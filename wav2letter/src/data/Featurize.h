/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <unordered_map>

#include "common/Dictionary.h"
#include "data/NumberedFilesLoader.h"
#include "feature/FeatureParams.h"
#include "feature/Sound.h"

namespace w2l {

typedef std::unordered_map<int, af::dim4> DimsMap;
typedef std::unordered_map<int, std::vector<int>> TargetFeatMap;

struct W2lFeatureData {
  std::vector<float> input; //unnormalized input
  TargetFeatMap targets; 
  af::dim4 inputDims; // T x K x FLAGS_channels x batchSz
  DimsMap targetDims;
  std::vector<int> sampleIds;
  af::dim4 sampleIdsDims;
  std::vector<float> inputFft; //raw complex fft input
  af::dim4 fftDims; // 2K x T x FLAGS_channels x batchSz
};

W2lFeatureData featurize(
    const std::vector<W2lLoaderData>& data,
    const DictionaryMap& dicts);

speech::FeatureParams defineSpeechFeatureParams();

int64_t getSpeechFeatureSize();

} // namespace w2l
