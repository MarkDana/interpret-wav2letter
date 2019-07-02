/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "module/module.h"
#include "runtime/Data.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"

using namespace w2l;
//zeroMode = 1 means use zero mode;
bool zeroMode = 0;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " [data_path] [dataset_name] [flags]");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }
    
  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  /* ===================== Create Dictionary ===================== */

  auto tokenDict = createTokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  auto wordDict = createWordDict(lexicon);
  LOG(INFO) << "Number of words: " << wordDict.indexSize();

  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* ===================== Create Dataset ===================== */
  // Load dataset
  int worldRank = 0;
  int worldSize = 1;
  auto ds = createDataset(FLAGS_test, dicts, lexicon, 1, worldRank, worldSize);

  ds->shuffle(3);
  int nSamples = ds->size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }
  LOG(INFO) << "[Dataset] Dataset loaded.";
  /* ===================== Test ===================== */
  TestMeters meters;

  EmissionSet emissionSet;
  meters.timer.resume();
  int cnt = 1;
  for (auto& sample:*ds){
  bool whether_threhold = true;
  auto rawinput = sample[kInputIdx];
  
  //to get prefft.txt
  std::ofstream preinput("/root/w2l/aboutFft/myfft.txt");
  if(preinput.is_open()){
  	preinput<<af::toString("myfft input:",rawinput);
  	preinput.close();
  }
  af::array finalinput;
   
  LOG(INFO)<<"rawinput 's dimension"<<rawinput.dims();
  int countzero=0;
    if(zeroMode)  //rawinput = raw audio
    {
	std::cout<<"read zero 's mode";

    	std::ifstream zerof("/root/wav2letter/50low.txt");
    	std::string line;
    	int row, col;
    	while(std::getline(zerof, line))
    	{
		countzero+=1;
		col = std::atoi(line.c_str());
         	std::getline(zerof, line);
         	row = std::atoi(line.c_str());
         	//LOG(INFO)<<"cols,raw "<<row<<' '<<col<<'\t';
		rawinput(row, col/2, af::span, af::span) = 0;
		//LOG(INFO)<<af::sum<float>(finalinput(row,col/2,af::span,af::span));
    	}
	zerof.close();
        //normalize after zeros
	auto mean = af::mean<float>(rawinput);
    	auto stdev = af::stdev<float>(rawinput);
        finalinput = (rawinput - mean) / stdev;
    }
    else  // rawinput = audio from zeros fft
    {    
    	auto mean = af::mean<float>(rawinput);
    	auto stdev = af::stdev<float>(rawinput);
    	finalinput = (rawinput - mean) / stdev;
    }
    std::cout<<"zeros number::"<<countzero<<std::endl;
	//edit @5.27
    auto rawEmission = network->forward({fl::input(finalinput)}).front();
	std::ofstream nowOutFile("/root/w2l/CTC/last_Test_Output.txt");
            if(nowOutFile.is_open())
            {
               nowOutFile<<af::toString("lastOutput is:", rawEmission.array());
               nowOutFile.close();
            }
	//endedit @5.27

    std::string emisspath = "/root/w2l/rawEmission.bin";
    W2lSerializer::save(emisspath, rawEmission);
    LOG(INFO) << "rawEmission norm is:" << af::norm(rawEmission.array()); 


    auto emission = afToVector<float>(rawEmission);
    auto ltrTarget = afToVector<int>(sample[kTargetIdx]);
    auto wrdTarget = afToVector<int>(sample[kWordIdx]);

    /* viterbiPath + remove duplication/blank */
    auto viterbiPath =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
    if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kAsgCriterion) {
      uniq(viterbiPath);
    }
    if (FLAGS_criterion == kCtcCriterion) {
      auto blankidx = tokenDict.getIndex(kBlankToken);
      viterbiPath.erase(
          std::remove(viterbiPath.begin(), viterbiPath.end(), blankidx),
          viterbiPath.end());
    }
    remapLabels(viterbiPath, tokenDict);
    remapLabels(ltrTarget, tokenDict);

    meters.lerSlice.add(viterbiPath, ltrTarget);

    auto wordViterbi = tknTensor2wrdTensor(
        viterbiPath, wordDict, tokenDict, tokenDict.getIndex(kSilToken));

    meters.werSlice.add(wordViterbi, wrdTarget);

    if (FLAGS_show) {
      meters.ler.reset();
      meters.wer.reset();
      meters.ler.add(viterbiPath, ltrTarget);
      meters.wer.add(wordViterbi, wrdTarget);

      std::cout << "|T|: " << tensor2letters(ltrTarget, tokenDict) << std::endl;
      std::cout << "|P|: " << tensor2letters(viterbiPath, tokenDict)
                << std::endl;
      std::cout << "[sample: " << cnt << ", WER: " << meters.wer.value()[0]
                << "\%, LER: " << meters.ler.value()[0]
                << "\%, total WER: " << meters.werSlice.value()[0]
                << "\%, total LER: " << meters.lerSlice.value()[0]
                << "\%, progress: " << static_cast<float>(cnt) / nSamples * 100
                << "\%]" << std::endl;
      ++cnt;
      if (cnt == FLAGS_maxload) {
        break;
      }
    }

    /* Save emission and targets */
    int N = rawEmission.dims(0);
    int T = rawEmission.dims(1);
    emissionSet.emissions.emplace_back(emission);
    emissionSet.letterTargets.emplace_back(ltrTarget);
    emissionSet.wordTargets.emplace_back(wrdTarget);

    // while testing we use batchsize 1 and hence ds only has 1 sampleid
    emissionSet.sampleIds.emplace_back(
        afToVector<std::string>(sample[kFileIdIdx]).front());

    emissionSet.emissionT.emplace_back(T);
    emissionSet.emissionN = N;
  }
  if (FLAGS_criterion == kAsgCriterion) {
    emissionSet.transition = afToVector<float>(criterion->param(0).array());
  }
  emissionSet.gflags = serializeGflags();

  meters.timer.stop();
  std::cout << "---\n[total WER: " << meters.werSlice.value()[0]
            << "\%, total LER: " << meters.lerSlice.value()[0]
            << "\%, time: " << meters.timer.value() << "s]" << std::endl;

  /* ====== Serialize emission and targets for decoding ====== */
  std::string cleanedTestPath = cleanFilepath(FLAGS_test);
  std::string savePath =
      pathsConcat(FLAGS_emission_dir, cleanedTestPath + ".bin");
  LOG(INFO) << "[Serialization] Saving into file: " << savePath;
  W2lSerializer::save(savePath, emissionSet);

  return 0;
}
/**/
