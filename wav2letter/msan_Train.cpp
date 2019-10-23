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
      int numNoise = 1000; //make 1000 noise sub-samples for the audio sample
      std::vector<float> Yloss(numNoise); //loss written into Yloss
      std::ofstream Yfile("/root/w2l/CTC/newDFT/loss.txt", std::ios::out);
      std::ofstream Y1("/root/w2l/CTC/newDFT/loss1.txt", std::ios::out);
      std::ofstream Y2("/root/w2l/CTC/newDFT/loss2.txt", std::ios::out);
  

    
      af::dim4 noiseDims = pre_sample[kFftIdx].dims(); //2K x T x FLAGS_channels x batchSz
      int T = noiseDims[1];
      int K = noiseDims[0]/2;
      af::array m = af::constant(1.0, af::dim4(K, T, noiseDims[2], noiseDims[3])); // Now m is K x T x FLAGS_channels x batchSz

      float mylr = 1.0;
      float lamda = 1.0;

      //pre_sample[kInputIdx] dims: T x K(257) x 1 x 1
      LOG_MASTER(INFO) << "pre_sample[kInputIdx] dims: " << pre_sample[kInputIdx].dims();
      //pre_sample[kFftIdx] dims: 2K(514) x T x 1 x 1
      LOG_MASTER(INFO) << "pre_sample[kFftIdx] dims: " << pre_sample[kFftIdx].dims();
      //LOG_MASTER(INFO) << af::toString("pre_sample fft's 6 values :", pre_sample[kFftIdx](af::seq(6)));
    
      std::ofstream preInput("/root/w2l/CTC/newDFT/preDft.txt");
      if(preInput.is_open())
      {
        preInput << af::toString("pre_dft values:",pre_sample[kInputIdx]);
        preInput.close();
      }
      
      const float inputMean=af::mean<float>(pre_sample[kInputIdx]);
      const float inputStdev=af::stdev<float>(pre_sample[kInputIdx]);
      LOG_MASTER(INFO) << "dft mean is:" << inputMean;//2136.15
      LOG_MASTER(INFO) << "dft stdev is:" << inputStdev;//5646.45
	  //the previous network's output f*
      fl::Variable preOutput; 
      //W2lSerializer::load("/root/w2l/rawEmission.bin", preOutput);
      auto prefinalinput=pre_sample[kInputIdx];
      af::array preStarInput= (prefinalinput-inputMean)/inputStdev;  // T x K x 1 x 1
      fl::Variable preTrueInput(preStarInput, false);
      ntwrk->eval();
      crit->eval();
      preOutput = ntwrk->forward({preTrueInput}).front();
      af::sync();

     
      std::ofstream preOutFile("/root/w2l/CTC/newDFT/preOutput.txt");
      if(preOutFile.is_open())
      {
	preOutFile << af::toString("preOutput is:", preOutput.array());
	preOutFile.close();
      }
      
      fl::Variable preRawInput(af::transpose(pre_sample[kInputIdx]), true);   // K x T
      
      auto mVar = fl::Variable(m, true);   // K x T x 1 x 1
      auto rang1Var = fl::Variable(af::range(af::dim4(K,T,K), 2), false);
      auto rang2Var = fl::Variable(af::range(af::dim4(K,T,K), 0), false);
      af::array r1 = af::range(af::dim4(K,T,K), 2);
      af::array r2 = af::range(af::dim4(K,T,K), 0);
      af::array rr = (r1 == r2);
      auto rangEq = fl::Variable(rr, false);

      ntwrk->train();
      crit->train();

      for (int i = 0; i < numNoise; i++) {

        LOG(INFO) << "=================noise sample " << i << "==================";
        af::sync();

        // m dispersion
        auto mTile = fl::tileAs(mVar, af::dim4(K,T,K));
        auto relu = fl::ReLU();
        auto out = relu(mTile - fl::abs(rang1Var - rang2Var)) /  (mTile * mTile);
        auto blar = fl::matmul(out, (mTile > 1)) + fl::matmul((mTile <= 1), rangEq);
        auto norm_index = fl::tileAs(fl::sum(blar, {2,3}), af::dim4(K,T,K));
        auto blared = blar / norm_index;
        auto inputTile = fl::tileAs(preRawInput, af::dim4(K,T,K));
        auto inputs = fl::moddims(fl::sum(inputTile * blared, {0}), af::dim4(T,K,1,1));

        auto meanIn = fl::tileAs(fl::mean(inputs, {0,1}), af::dim4(T,K,1,1));
        auto stdevIn = fl::tileAs(fl::sqrt(fl::var(inputs, {0,1})), af::dim4(T,K,1,1));

        auto realInput = (inputs - meanIn) / stdevIn;
        
        if (af::anyTrue<bool>(af::isNaN(realInput.array()))) {
          LOG(FATAL) << "real Input has NaN values";
        }

        // forward
        auto output = ntwrk->forward({realInput}).front();

        std::ofstream nowOutFile("/root/w2l/CTC/newDFT/lastOutput.txt");
	if(nowOutFile.is_open())
	{
	   nowOutFile<<af::toString("lastOutput is:", output.array());
	   nowOutFile.close();
	}

        
        auto f_L2 = fl::norm(output - preOutput, {0,1,2,3});
        auto loss1 = f_L2 * f_L2;
        auto loss2 = - lamda * fl::sum(fl::log(mVar * mVar), {0,1,2,3});
        auto myloss = loss1 + loss2;
        //auto firloss = fl::MeanSquaredError();
        //auto myloss = firloss(output, preOutput);

        float totloss = myloss.scalar<float>();

        LOG(INFO) << "f star norm is:" << af::norm(preOutput.array());
        LOG(INFO) << "f now norm is:" << af::norm(output.array());
        Yfile << totloss << std::endl;
        Y1 << loss1.scalar<float>() << std::endl;
        Y2 << loss2.scalar<float>() <<std::endl;

        af::sync();
       

        if (af::anyTrue<bool>(af::isNaN(myloss.array()))) {
          LOG(FATAL) << "Loss has NaN values";
        }

        //clear the gradients for next iteration
        netopt->zeroGrad();
        critopt->zeroGrad();

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
        //update parameter mVar
	mVar.array() = mVar.array() - mylr * mVar.grad().array();  
              
      }
	  
      af::sync();
      //network params whether to be changed
      fl::MSEMeter mymeter;
      auto psize = ntwrk->params().size();
      for(int j=0 ; j<psize; j++) {
	 mymeter.add(ntwrk->param(j).array(), pretrained_params[j].array());
      }
      LOG(INFO) << "the network params change " << mymeter.value();  
	

      if (FLAGS_reportiters == 0) {
      // if (0 == 0) {
        //runValAndSaveModel(curEpoch, netopt->getLr(), critopt->getLr());
        //std::string mpath = "/root/w2l/aboutM/last_m.bin";
        //W2lSerializer::save(mpath, m);

        std::ofstream mfile("/root/w2l/CTC/newDFT/lastm.txt");
        if(mfile.is_open())
        {
          mfile << af::toString("last m is:", mVar.array());
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





