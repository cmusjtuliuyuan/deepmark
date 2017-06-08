-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 20) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 10) / 2 + 1) -- conv2
    return sizes
end


local function RNNModule(inputDim, hiddenDim)
    require 'rnn'
    return nn.SeqBRNN(inputDim, hiddenDim)
end

local function cudnnDeepSpeech2(miniBatchSize, freqBins, nGPUs, useOptnet)

    local conv = nn.Sequential()
    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    conv:add(nn.SpatialConvolution(1, 1152, 11, 13, 3, 1))
    conv:add(nn.Clamp(0, 20))

    conv:add(nn.View(1152, -1):setNumInputDims(3)) -- batch x features x seqLength
    conv:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    local rnns = nn.Sequential()
    local rnnModule = RNNModule(1152, 1152)
    rnns:add(rnnModule:clone())
    rnnModule = RNNModule(1152, 1152)

    for i = 1, 9 do
        rnns:add(nn.Bottle(nn.BatchNormalization(1152), 2))
        rnns:add(rnnModule:clone())
    end

    local fullyConnected = nn.Sequential()
    fullyConnected:add(nn.BatchNormalization(1152))
    fullyConnected:add(nn.Linear(1152, 29))

    local model = nn.Sequential()
    model:add(conv)
    model:add(rnns)
    model:add(nn.Bottle(fullyConnected, 2))
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    return model, 'cudnnDeepSpeech2', calculateInputSizes
end

return cudnnDeepSpeech2
