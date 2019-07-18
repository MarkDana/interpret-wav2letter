/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "cstdio"

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include "data/W2lDataset.h"
#include "data/W2lNumberedFilesDataset.h"

using namespace w2l;


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + std::string() +
      " continue [directory] [flags]\n or " + std::string(argv[0]) +
      " fork [directory/model] [flags]");

  /* ===================== Parse Options ===================== */
  int runIdx = 1; // current #runs in this path
  std::string runPath; // current experiment path
  std::string reloadPath; // path to model to reload
  std::string runStatus = argv[1];
  int startEpoch = 0;

  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  std::vector<fl::Variable> pretrained_params;

  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  if (runStatus == "fork") {
    reloadPath = argv[2];
    /* ===================== Create Network ===================== */
    LOG(INFO) << "Network reading pre-trained model from " << reloadPath;
    W2lSerializer::load(reloadPath, cfg, network, criterion);
    pretrained_params = network->params();

    //pre-trained network architecture
    LOG(INFO) << "[Network] is " << network->prettyString();
    LOG(INFO) << "[Criterion] is " << criterion->prettyString();
    LOG(INFO) << "[Network] params size is " << network->params().size();
    LOG(INFO) << "[Network] number of params is " << numTotalParams(network);

    //pre-trained network flags
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }

    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    if (argc > 3) {
      LOG(INFO) << "Parsing command line flags";
      LOG(INFO) << "Overriding flags should be mutable when using `fork`";
      gflags::ParseCommandLineFlags(&argc, &argv, false);
    }

    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file" << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  af::setMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);

  maybeInitDistributedEnv(
      FLAGS_enable_distributed,
      FLAGS_world_rank,
      FLAGS_world_size,
      FLAGS_rndv_filepath);

  auto worldRank = fl::getWorldRank();
  auto worldSize = fl::getWorldSize();

  bool isMaster = (worldRank == 0);



  LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");


  LOG_MASTER(INFO) << "Experiment path: " << runPath;
  LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;


  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, getEnvVar("USER")},
      {kHostName, getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  /* ===================== Create Dictionary & Lexicon ===================== */
  Dictionary dict = createTokenDict();
  int numClasses = dict.indexSize();
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  LexiconMap lexicon;
  if (FLAGS_listdata) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  }

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  // network, criterion have been loaded before
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
  if (runStatus == "train" || runStatus == "fork") {
    netoptim = initOptimizer(
        network, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer(criterion, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  }
  LOG_MASTER(INFO) << "[Network Optimizer] " << netoptim->prettyString();
  LOG_MASTER(INFO) << "[Criterion Optimizer] " << critoptim->prettyString();

  printf("ok runpath is %s\n",runPath.c_str());
  //07-10-上午：这行可以正常输出，但LOG只能打到119行，why

  /* ===================== Meters ===================== */
  

  /* ===================== Logging ===================== */
  

  /* ===================== Create Dataset ===================== */
  auto trainds = createDataset(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);


  /* ===================== Hooks ===================== */

  double gradNorm = 1.0 / (FLAGS_batchsize * worldSize);

  auto train = [gradNorm,
                pretrained_params,
                &startEpoch](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<W2lDataset> trainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   std::shared_ptr<fl::FirstOrderOptimizer> critopt,
                   double initlr,
                   double initcritlr,
                   bool clampCrit,
                   int nepochs) {
    fl::distributeModuleGrads(ntwrk, gradNorm);
    fl::distributeModuleGrads(crit, gradNorm);

    // synchronize parameters across processes
    fl::allReduceParameters(ntwrk);
    fl::allReduceParameters(crit);


    int64_t curEpoch = startEpoch;
    int64_t sampleIdx = 0;
    while (curEpoch < nepochs) {
      double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
      netopt->setLr(lrScale * initlr);
      critopt->setLr(lrScale * initcritlr);

      ++curEpoch;
      ntwrk->train();
      crit->train();

      af::sync();
      
      LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";

      //the size of trainset is just 1.
      auto pre_sample = trainset->get(0); //make noises for one audio sample
      int numNoise = 300000; //make 1000 noise sub-samples for the audio sample
      std::vector<float> Yloss(numNoise); //loss written into Yloss
      std::ofstream Yfile("/root/w2l/CTC/loss.txt", std::ios::out);
      std::ofstream Mmeanfile("/root/w2l/CTC/m_mean.txt", std::ios::out);
      std::ofstream Mvarfile("/root/w2l/CTC/m_var.txt", std::ios::out);
      std::ofstream Mlossfile("/root/w2l/CTC/m_loss.txt", std::ios::out);
      std::ofstream mylossfile("/root/w2l/CTC/myloss.txt", std::ios::out);
      std::ofstream myloss_grad_mean_file("/root/w2l/CTC/myloss_grad_mean.txt", std::ios::out);
      std::ofstream myloss_grad_var_file("/root/w2l/CTC/myloss_grad_var.txt", std::ios::out);
      std::ofstream mloss_grad_mean_file("/root/w2l/CTC/mloss_grad_mean.txt", std::ios::out);
      std::ofstream mloss_grad_var_file("/root/w2l/CTC/mloss_grad_var.txt", std::ios::out);
      std::ofstream fft_mean_file("/root/w2l/CTC/fft_mean.txt", std::ios::out);


      
      
      //std::vector<float> firGradnorm(numNoise);
      //std::ofstream firGradnormFile("/root/w2l/aboutM/firGradnorm.txt", std::ios::out);
      //std::vector<float> secGradnorm(numNoise);
      //std::ofstream secGradnormFile("/root/w2l/aboutM/secGradnorm.txt", std::ios::out);
      //std::vector<float> totGradnorm(numNoise);
      //std::ofstream totGradnormFile("/root/w2l/aboutM/totGradnorm.txt", std::ios::out);

      af::dim4 noiseDims = pre_sample[kFftIdx].dims(); //2K x T x FLAGS_channels x batchSz
      int T = noiseDims[1];
      int K = noiseDims[0]/2;
      auto m = af::constant(1.0, af::dim4(K, T, noiseDims[2], noiseDims[3])); // Now m is K x T x FLAGS_channels x batchSz

      // auto m = af::constant(0.1, noiseDims);
      //auto m = af::constant(0.1,noiseDims);
      //auto m=fl::normal(noiseDims,0.002,0.1).array();
      // float mylr = 0.001;
      float mylr = 100000.0;

      //the previous network's output f*
      fl::Variable preOutput; 
      //W2lSerializer::load("/root/w2l/rawEmission.bin", preOutput);

      //pre_sample[kInputIdx] dims: T x K(257) x 1 x 1
      LOG_MASTER(INFO) << "pre_sample[kInputIdx] dims: " << pre_sample[kInputIdx].dims();
      //pre_sample[kFftIdx] dims: 2K(514) x T x 1 x 1
      LOG_MASTER(INFO) << "pre_sample[kFftIdx] dims: " << pre_sample[kFftIdx].dims();
      const float fftmean = af::mean<float>(pre_sample[kFftIdx]);
      const float fftstdev = af::stdev<float>(pre_sample[kFftIdx]);
      //LOG_MASTER(INFO) << af::toString("pre_sample fft's 6 values :", pre_sample[kFftIdx](af::seq(6)));
      LOG_MASTER(INFO) << "fft mean is:" << af::mean<float>(pre_sample[kFftIdx]);//-0.12
      LOG_MASTER(INFO) << "fft stdev is:" << af::stdev<float>(pre_sample[kFftIdx]);//4268.81
      //LOG_MASTER(INFO) << "dft mean is:" << af::mean<float>(pre_sample[kInputIdx]);//2136.15
      //LOG_MASTER(INFO) << "dft stdev is:" << af::stdev<float>(pre_sample[kInputIdx]);//5646.45

      std::ofstream preinput("/root/w2l/CTC/preFft.txt");
      if(preinput.is_open())
      {
        preinput << af::toString("pre_fft values:",pre_sample[kFftIdx]);
        preinput.close();
      }
      //Notice:here prefft is 2K*T
      //Notice:but maskMusic is K*T



      //using network to generate preOutput 
      auto prefinalinput=pre_sample[kInputIdx];
      const float inputmean=af::mean<float>(pre_sample[kInputIdx]);
      const float inputstdev=af::stdev<float>(pre_sample[kInputIdx]);
      prefinalinput= (prefinalinput-inputmean)/inputstdev;
      fl::Variable pretruefinalinput(prefinalinput,false);
  
      ntwrk->eval();
      crit->eval();
      preOutput = ntwrk->forward({pretruefinalinput}).front();
      auto preOutput_arr=preOutput.array();
      af::sync();

      af::dim4 outputDims = preOutput_arr.dims();
      int tokendim=outputDims[0];
      std::vector<int> axes{1};
      fl::Variable tmpOutput = fl::sqrt(fl::var(preOutput,axes));
      tmpOutput = fl::tileAs(tmpOutput, preOutput);
      fl::Variable addpreOutput = preOutput/tmpOutput;
  //for (size_t i = 0; i < tokendim; i=i+1)
      //{
        //  auto framestdev=af::stdev<float>(preOutput_arr(i,af::span,af::span,af::span));
          //preOutput_arr(i,af::span,af::span,af::span)=preOutput_arr(i,af::span,af::span,af::span)/framestdev;
          
      //}


      std::ofstream preOutFile("/root/w2l/CTC/preOutput.txt");
      if(preOutFile.is_open())
      {
  preOutFile << af::toString("preOutput is:", preOutput_arr);
  preOutFile.close();
      }

      // af::array zerowgt = af::identity(31,31);
      // zerowgt(0, 0) = 0;
      // zerowgt(1, 1) = 0;
  
      //   zerowgt(28, 28) = 0;
      //   zerowgt(29, 29) = 0;
      //   zerowgt(30, 30) = 0;

      // fl::Variable zeroweight(zerowgt, true);
  // auto addpreOutput = fl::Variable(af::matmul(zerowgt, preOutput.array()), false);
        //fl::Variable addpreOutput(preOutput_arr,true);
        //auto softmax_preOutput = fl::softmax(addpreOutput,1);
  // ignore 5 dimensions, softmax rest dimensions
  //auto tmpout = softmax_preOutput(af::seq(2, 27), af::span, af::span, af::span);
  auto tmpout = addpreOutput(af::seq(2, 27), af::span, af::span, af::span);
  auto softmax_add_preOutput = fl::softmax(tmpout, 0);
  //softmax_preOutput(af::seq(2,27),af::span,af::span,af::span)=softmax_tmpOut;
  // addpreOutput(af::seq(2,27),af::span,af::span,af::span)=softmax_tmpOut;
//  auto softmax_tmpOut = fl::softmax(tmpout,1);
//  auto softmax_preOut = fl::tileAs(softmax_tmpOut, softmax_preOutput.array().dims());
  // auto softmax_add_preOutput = fl::matmul(zeroweight, softmax_preOutput);
  // auto softmax_add_preOutput = fl::matmul(zeroweight, addpreOutput);



      std::ofstream preOutFile_0("/root/w2l/CTC/preOutput_0.txt");
      if(preOutFile_0.is_open())
      {
  preOutFile_0 << af::toString("preOutput_0 is:", softmax_add_preOutput.array());
  preOutFile_0.close();
      }
      
      
      ntwrk->train();
      crit->train();


      ///////////////////////////////////////////////////////////////////////////////////////////////
      auto rawinput = pre_sample[kFftIdx];
      af::array absinput(af::dim4(K, T, noiseDims[2], noiseDims[3]));
      auto Z_add = af::constant (0,af::dim4(K, T, K, noiseDims[3])); // Z_add is Z
      auto Z_grad = af::constant (0,af::dim4(K, T, K, noiseDims[3])); // Z_grad is partial(Z_pji)/partial(m_p_j)
      af::array absinput_after_blur(af::dim4(K, T, noiseDims[2], noiseDims[3]));

      for (size_t j = 0; j < 2*K; j=j+2)
        {
            auto fir = rawinput(j, af::span, af::span, af::span);
            //LOG(INFO) << "fir row(i) dims is :" << fir.array().dims() << " " << af::toString("row(i) first value is ", fir.array()(0));
            auto sec = rawinput(j+1, af::span, af::span, af::span);
            //note shallow copy in fl::Variable
            auto temp = af::sqrt(fir * fir + sec * sec);
            absinput(j/2, af::span, af::span, af::span) =  temp;
        }


      for (int i = 0; i < numNoise; i++) {

        LOG(INFO) << "=================noise sample " << i << "==================";
        // meters
        af::sync();

        // if (i>30000){mylr=100;}
        // else if (i>10000){mylr=500;}
        // else if (i>5000){mylr=1000;}
        
        if (af::anyTrue<bool>(af::isNaN(rawinput)) ||
            af::anyTrue<bool>(af::isNaN(rawinput))) {
          LOG(FATAL) << "pre_sample has NaN values";
        }
//////////////////////////////////////////////////////////////////////////////////////////////////////
  //auto epsilon = (af::randn(noiseDims)) * 4268; 
  //       auto epsilon = fl::normal(noiseDims,fftstdev,0).array(); //add noises
  // LOG(INFO)<<"epsilon mean is:"<<af::mean<float>(epsilon);
  // LOG(INFO)<<"epsilon stdev is:"<<af::stdev<float>(epsilon);
  // //save last iter epsilon parameter:
  // if (i == numNoise-1)
  // {
  //    std::ofstream epsfile("/root/w2l/CTC/epsilon.txt");
  //    if(epsfile.is_open())
  //      {
  //       epsfile << af::toString("epsilon values:", epsilon);
  //             epsfile.close();
  //    }
  // }



  //LOG(INFO)<<af::toString("epsilon 6 values:", epsilon(af::seq(6)));
  //LOG(INFO)<<af::toString("m 6 values:", m(af::seq(6)));
  //LOG(INFO)<<af::toString("rawinput 6 values:",rawinput(af::seq(6)));
        
        
        //LOG(INFO) << "m_epsilon mean :" << af::mean<float>(m*epsilon);
        //LOG(INFO) << "m_epsilon stdev :" << af::stdev<float>(m*epsilon);
      

// for (size_t iloop = 0; iloop < K; ++iloop){
//   for (size_t jloop = 0; jloop < T; ++jloop){
//     absinput_after_blur(iloop,jloop,af::span,af::span) = absinput(iloop,jloop,af::span,af::span);

    
//     // gfor (af::seq ploop, std::max(iloop-int(m_p_j),0), std::min(iloop+int(m_p_j),K-1)){
//     gfor (af::seq ploop, K){
//       auto m_p_j = af::moddims(m(ploop,jloop,0,0),K); // dim of K*1*1*1
//       auto m_floor = af::floor(m_p_j); // dim of K*1*1*1

//       auto sum_m_p_j=m_floor*(2*m_p_j-m_floor-1) + m_p_j; // dim of K*1*1*1
//       auto sum_mpj_partial_to_mpj=2*m_p_j; // dim of K*1*1*1

//       //这里只看 abs(ploop-iloop)<m_p_j 的部分
//       auto condition1 = (af::abs(ploop-iloop)<m_p_j);
//       auto condition2 = ((ploop - iloop)==0);


//       auto Z_add_pji = condition1.as(f32) * ((!condition2).as(f32) * (af::moddims(absinput(ploop,jloop,0,0),K)*(m_p_j-af::abs(iloop-ploop))/sum_m_p_j) + condition2.as(f32) * (af::moddims(absinput(ploop,jloop,0,0),K)*(m_p_j-sum_m_p_j)/sum_m_p_j));
//       auto Z_grad_pji = condition1.as(f32) * ((!condition2).as(f32) * (af::moddims(absinput(ploop,jloop,0,0),K)*(sum_m_p_j - sum_mpj_partial_to_mpj*(m_p_j-abs(iloop-ploop)))/(sum_m_p_j*sum_m_p_j)) + condition2.as(f32) * (af::moddims(absinput(ploop,jloop,0,0),K)*((1-sum_mpj_partial_to_mpj)*sum_m_p_j-sum_mpj_partial_to_mpj*(m_p_j-sum_m_p_j))/(sum_m_p_j*sum_m_p_j)));
      

//       Z_add(ploop,jloop,iloop,af::span) = Z_add_pji;
//       Z_grad(ploop,jloop,iloop,af::span) = Z_grad_pji;
//     } 

//     absinput_after_blur(iloop,jloop,af::span)+=af::sum(Z_add(af::span,jloop,iloop),0);
//   }
//   // printf("i=%d\n",iloop);
// } 

  //As above, this method is too slow, use tile to parallel this part, as follows


        absinput_after_blur(af::span,af::span,af::span,af::span) = absinput(af::span,af::span,af::span,af::span);

        // m = af::max(m,af::constant(0, m.dims()));

        m = af::abs(m);

        af::array MTiled = af::tile(m, af::dim4(1, 1, K));
        af::array absTiled = af::tile(absinput, af::dim4(1, 1, K));
        af::array iloop = af::range(af::dim4(K, T, K), 2);
        af::array ploop = af::range(af::dim4(K, T, K), 0);

        af::array i_e_p = (iloop == ploop);
        af::array cond  = af::abs(iloop - ploop) < MTiled;

        auto m_floor = af::floor(MTiled);
        auto sum_m_p_j=m_floor*(2*MTiled-m_floor-1) + MTiled;
        auto sum_mpj_partial_to_mpj=2*MTiled;

        //////////////这种写法是将越界的全部返回加起来，现在用的方法是越界后再归一化////////////
        // af::array f1_1 = absTiled*(MTiled-af::abs(iloop-ploop))/sum_m_p_j; //i!=p, add
        // af::array f1_2 = absTiled*(sum_m_p_j - sum_mpj_partial_to_mpj*(MTiled-abs(iloop-ploop)))/(sum_m_p_j*sum_m_p_j); //i!=p, grad

        // Z_add = cond * ((1 - i_e_p) * f1_1);
        // Z_grad = cond * ((1 - i_e_p) * f1_2);

        // af::array f2_1 = (-1.0)*af::tile(af::sum(Z_add, 2), af::dim4(1, 1, K)); //i==p, add
        // af::array f2_2 = (-1.0)*af::tile(af::sum(Z_grad, 2), af::dim4(1, 1, K)); //i==p, grad

        // Z_add += cond * i_e_p * f2_1;
        // Z_grad += cond * i_e_p * f2_2;

        // absinput_after_blur += af::transpose(af::moddims(af::sum(Z_add,0), af::dim4(T, K, 1, 1)));
        ////////////////////////////////////////////////////////////////////////////

        af::array f1_1 = absTiled*(MTiled-af::abs(iloop-ploop))/sum_m_p_j; //i!=p, add
        af::array f1_2 = absTiled*(sum_m_p_j - sum_mpj_partial_to_mpj*(MTiled-abs(iloop-ploop)))/(sum_m_p_j*sum_m_p_j); //i!=p, grad

        af::array original_ratio_to_nowsum = absTiled/af::tile(af::sum(cond * f1_1,2),af::dim4(1, 1, K));
        f1_1 *= original_ratio_to_nowsum;
        f1_2 *= original_ratio_to_nowsum;

        af::array f2_1 = absTiled*(original_ratio_to_nowsum*MTiled-sum_m_p_j)/sum_m_p_j; //i==p, add
        af::array f2_2 = absTiled*((original_ratio_to_nowsum-sum_mpj_partial_to_mpj)*sum_m_p_j-sum_mpj_partial_to_mpj*(original_ratio_to_nowsum*MTiled-sum_m_p_j))/(sum_m_p_j*sum_m_p_j); //i==p, grad

        Z_add = cond * (i_e_p * f2_1 + (1 - i_e_p) * f1_1);
        Z_grad = cond * (i_e_p * f2_2 + (1 - i_e_p) * f1_2);
                
        absinput_after_blur += af::transpose(af::moddims(af::sum(Z_add,0), af::dim4(T, K, 1, 1)));


        //Notice:here prefft is 2K*T
        //Notice:but maskMusic is K*T, and angle remains still
        if((i+1)%(numNoise/10) == 0)
        {
            char outdir[80];

            sprintf(outdir, "/root/w2l/CTC/music_mask_%d.txt", i);
        
            std::ofstream fft_mask_now(outdir);
            if(fft_mask_now.is_open())
            {
               fft_mask_now<<af::toString("mask music is:", absinput_after_blur);
               fft_mask_now.close();
            }

            char m_dir[80];

            sprintf(m_dir, "/root/w2l/CTC/m_%d.txt", i);
        
            std::ofstream m_now(m_dir);
            if(m_now.is_open())
            {
               m_now<<af::toString("m is:", m);
               m_now.close();
            }
        }

        // if(i == 0)
        // {
        
        //     // std::ofstream debug_zadd("/root/w2l/CTC/debug_zadd.txt");
        //     // if(debug_zadd.is_open())
        //     // {
        //     //    debug_zadd<<af::toString("debug_zadd is:", Z_add);
        //     //    debug_zadd.close();
        //     // }

        //     // std::ofstream debug_zgrad("/root/w2l/CTC/debug_zgrad.txt");
        //     // if(debug_zgrad.is_open())
        //     // {
        //     //    debug_zgrad<<af::toString("debug_zgrad is:", Z_grad);
        //     //    debug_zgrad.close();
        //     // }

        //     std::ofstream realadd("/root/w2l/CTC/realadd.txt");
        //     if(realadd.is_open())
        //     {
        //        realadd<<af::toString("realadd is:", af::transpose(af::moddims(af::sum(Z_add,0), af::dim4(T, K, 1, 1))));
        //        realadd.close();
        //     }


        // }


        fft_mean_file << af::mean<float>(absinput_after_blur) << std::endl;

        //T x K x FLAGS_channels x batchSz
        // af::array trInput = af::transpose(absinput);

        af::array trInput = af::transpose(absinput_after_blur);
        // printf("trInput okok\n");

        // dft kInputIdx not normalized
        //LOG(INFO) << "dft abs mean :" << af::mean<float>(absinput);
        //LOG(INFO) << "dft abs stdev :" << af::stdev<float>(absinput);

        // normalization
        auto mean = af::mean<float>(trInput); // along T and K two dimensions 1x1x1x1
        auto stdev = af::stdev<float>(trInput); //1 scalar
        auto finalInput = (trInput - mean) / stdev;
        fl::Variable trueInput(finalInput, true);
        
        auto indif = af::mean<float>(trInput - pre_sample[kInputIdx]);
        LOG(INFO) << "dft input difference mean is:" << indif;
        /*
        std::ofstream exfile("/home/zd/beforenorm.txt");
        if(exfile.is_open())
        {  
           exfile << af::toString("before norm", finalInput.array());
           exfile.close();
        }
        */

        // forward
        auto output = ntwrk->forward({trueInput}).front();
  auto output_arr = output.array();
  int tokendim=outputDims[0]; 
  
  std::vector<int> axes1{1};
  fl::Variable tmpaddOutput = fl::sqrt(fl::var(output, axes1));
  tmpaddOutput = fl::tileAs(tmpaddOutput, output);
  fl::Variable addoutput = output/tmpaddOutput;

     //   for (size_t j = 0; j < tokendim; j=j+1)
      //{
        //    auto framestdev=af::stdev<float>(output_arr(j,af::span,af::span,af::span));  
  //  output_arr(j,af::span,af::span,af::span)=output_arr(j,af::span,af::span,af::span)/framestdev;
          
      //}

        std::ofstream nowOutFile("/root/w2l/CTC/lastOutput.txt");
      if(nowOutFile.is_open())
      {
         nowOutFile<<af::toString("lastOutput is:", output_arr);
         nowOutFile.close();
      }
 //        af::array wgt = af::identity(31, 31); // numClasses are 31 tokens
  // wgt(0, 0) = 0;
 //        wgt(1, 1) = 0;
       
 //        wgt(28, 28) = 0;
 //        wgt(29, 29) = 0;
 //        wgt(30, 30) = 0;
 //        auto addweight = fl::Variable(wgt, true);
        // auto addoutput = fl::matmul(addweight, output);
        //fl::Variable addoutput(output_arr,true);
  // auto softmax_output = fl::softmax(addoutput,1);
  auto tmp = addoutput(af::seq(2,27),af::span,af::span,af::span);
  auto softmax_add_output = fl::softmax(tmp,0);
  // addoutput(af::seq(2,27),af::span,af::span,af::span)=softmax_tmp; 
  // auto softmax_add_output = fl::matmul(addweight, addoutput);

        af::sync();
  if(i == numNoise-1)
  {
      std::ofstream nowOutFile_0("/root/w2l/CTC/lastOutput_0.txt");
      if(nowOutFile_0.is_open())
      {
         nowOutFile_0<<af::toString("lastOutput_0 is:", softmax_add_output.array());
         nowOutFile_0.close();
      }
  }
        
        //LOG(INFO) << "network forward output dims is "<< output.array().dims();
        //LOG(INFO) << "load rawEmission preOutput dims is :" << preOutput.array().dims() ;

  // printf("backward okok\n");

       float lambda = 1e-12;
        //float lambda = 100;
        auto f_L2 = fl::norm(softmax_add_preOutput - softmax_add_output, {0,1});
        auto m_entropy = af::sum<float> (af::log(af::abs(m))); 
        auto myloss = f_L2 * f_L2;
        float m_mean=af::mean<float>(m);
        float m_var=af::var<float>(m);
  
        //auto firloss = fl::MeanSquaredError();
        //auto myloss = firloss(output, preOutput);

        float totloss = myloss.scalar<float>() - lambda * m_entropy;
        // float totloss = lambda * myloss.scalar<float>() - m_entropy;

        LOG(INFO) << "f star norm is:" << af::norm(preOutput.array());
        LOG(INFO) << "f now norm is:" << af::norm(output.array());
        LOG(INFO) << "loss - f difference is :" << myloss.scalar<float>();
        LOG(INFO) << "loss - logm is :" << m_entropy;
        LOG(INFO) << "loss is:" << totloss;
        printf("now training m%d\ttotloss=%f\tmmean=%f\n",i,totloss,m_mean);
        Yfile << totloss << std::endl;
        Mlossfile << m_entropy << std::endl;
        Mmeanfile << m_mean<<std::endl;
        Mvarfile << m_var<<std::endl;
        mylossfile << myloss.scalar<float>()<<std::endl;


        af::sync();
       

        if (af::anyTrue<bool>(af::isNaN(myloss.array()))) {
          LOG(FATAL) << "Loss has NaN values";
        }

        // clear the gradients for next iteration
        netopt->zeroGrad();
        critopt->zeroGrad();
 //        zeroweight.zeroGrad();
  // addweight.zeroGrad();

        //Compute gradients using backprop
        myloss.backward();
        af::sync();
  //Print output's Grad
  if(i == numNoise-1)
  {
            std::ofstream outputGradFile("/root/w2l/CTC/outputGrad.txt");
      if(outputGradFile.is_open())
      {
          outputGradFile << af::toString("output Grad is:", output.grad().array());
                outputGradFile.close();
      }
  }

        if (FLAGS_maxgradnorm > 0) {
          auto params = ntwrk->params();
          if (clampCrit) {
            auto critparams = crit->params();
            params.insert(params.end(), critparams.begin(), critparams.end());
          }
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }

        //critopt.step();
        //netopt.step();
        //update parameter m

        auto sigma2 = stdev * stdev;
        auto dy = trueInput.grad().array(); //T x K
        auto dsigma2 = af::sum<float>(dy * (trInput - mean) * (-0.5) * std::pow(sigma2, -1.5));
        auto dmu = af::sum<float>(dy * (-1.0/std::pow(sigma2, 0.5))) + af::sum<float>(-2 * (trInput - mean)) * dsigma2 / (T * K);
        auto dx = dy / std::pow(sigma2, 0.5) + dsigma2 * 2 * (trInput - mean) / (T * K) + dmu / (T * K); 
        af::array xGrad = af::transpose(dx); // K x T x 1 x 1
        af::array mGrad = af::constant(0, af::dim4(K, T, 1, 1));

        //xGrad is ∂ myloss / ∂ absinput_after_blur;

        // printf("xGrad okok\n");

        // af::dim4 tmpcout = mGrad.dims();
        // printf("mGrad is %dx%dx%dx%d\n",tmpcout[0],tmpcout[1],tmpcout[2],tmpcout[3]);

        // for (size_t igrad=0; igrad<K; ++igrad){
        //   mGrad(igrad,af::span,af::span,af::span) = af::sum(xGrad*Z_grad(af::span,af::span,igrad,af::span),0);
        // }

        mGrad = af::transpose(af::moddims(af::sum(af::tile(xGrad, af::dim4(1, 1, K))*Z_grad,0),af::dim4(T, K, 1, 1)));

        // printf("mGrad okok\n");

        auto mGrad_aboutm_entropy = 1 / m ;

        myloss_grad_mean_file << af::mean<float>(mGrad)<<std::endl;
        myloss_grad_var_file << af::var<float>(mGrad)<<std::endl;
        mloss_grad_mean_file << af::mean<float>(mGrad_aboutm_entropy)<<std::endl;
        mloss_grad_var_file << af::var<float>(mGrad_aboutm_entropy)<<std::endl;


        mGrad =  mGrad - lambda * mGrad_aboutm_entropy;

        m = m - mylr * mGrad;
        
        
        //network params whether to be changed
        fl::MSEMeter mymeter;
        auto psize = ntwrk->params().size();
        for(int j=0 ; j<psize; j++) {
          mymeter.add(ntwrk->param(j).array(), pretrained_params[j].array());
        }
        LOG(INFO) << "the network params change " << mymeter.value();        
      }


      af::sync();

      if (FLAGS_reportiters == 0) {
      // if (0 == 0) {
        //runValAndSaveModel(curEpoch, netopt->getLr(), critopt->getLr());
        //std::string mpath = "/root/w2l/aboutM/last_m.bin";
        //W2lSerializer::save(mpath, m);

        std::ofstream mfile("/root/w2l/CTC/lastm.txt");
        if(mfile.is_open())
        {
          mfile << af::toString("last m is:", m);
          mfile.close();
        }
      }
    }
  };

  /* ===================== Train ===================== */
  train(
      network,
      criterion,
      trainds,
      netoptim,
      critoptim,
      FLAGS_lr,
      FLAGS_lrcrit,
      true /* clampCrit */,
      FLAGS_iter);

  LOG_MASTER(INFO) << "Finished my training";
  return 0;
}


//M shape is K * T * 1 * 1, float
//abs shape is K * T * 1 * 1, float

//Z1 shape is K * T * K * 1, initially all 0
//Z2 shape is K * T * K * 1, initially all 0

//intend to assign 3D Z1, Z2 based on 2D M, abs
//f*_* are functions with args pfloat, returning float




