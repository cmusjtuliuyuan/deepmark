-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 20) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 10) / 2 + 1) -- conv2
    return sizes
end

local function cudnnDeepSpeech2(miniBatchSize, freqBins, nGPUs, useOptnet)

    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(1, 1152, 11, 13, 3, 1))
    model:add(cudnn.SpatialBatchNormalization(1152))
    model:add(cudnn.ClippedReLU(20, true))

    model:add(nn.View(1152*((freqBins-13)/1+1), -1):setNumInputDims(3)) -- batch x features x seqLength
    model:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    model:add(cudnn.BatchBRNNReLU(1152*((freqBins-13)/1+1), 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))
    model:add(cudnn.BatchBRNNReLU(1152, 1152))

    model:add(nn.Bottle(nn.Linear(1152, 1152))) -- keeps the output 3D for multi-GPU.
    model:add(nn.Bottle(nn.Linear(1152, 29)))
    model:cuda()
    if useOptnet then
	local optnet = require 'optnet'
	local seqLength = 100
	local sampleInput = torch.zeros(2,1,freqBins, seqLength):cuda()
--[[        local output = model:updateOutput(sampleInput)
        model:backward(sampleInput, output)
        cutorch.synchronize()
	mem1=optnet.countUsedMemory(model) --]]
        optnet.optimizeMemory(model, sampleInput, {inplace=false, mode = 'training'})
--[[        model:backward(sampleInput, output)
	collectgarbage()
	mem2=optnet.countUsedMemory(model)
        print('Before optimization        : ', mem1.gradInputs)
        print('After optimization         : ', mem2.gradInputs)		--]]
    end
    model = makeModelParallel(model, nGPUs)
    return model, 'cudnnDeepSpeech2', calculateInputSizes
end

function makeModelParallel(model, nGPU)
    if nGPU >= 1 then
        if nGPU > 1 then
            gpus = torch.range(1, nGPU):totable()
            dpt = nn.DataParallelTable(1, true, true):add(model, gpus):threads(function()
                local cudnn = require 'cudnn'
                cudnn.fastest = true
                require 'BatchBRNNReLU'
                require 'rnn'
            end)
            dpt.gradInput = nil
            model = dpt
        end
--        model:cuda()
    end
    return model
end

return cudnnDeepSpeech2
