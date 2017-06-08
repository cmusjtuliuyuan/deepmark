require 'sys'
require 'Dataset'
require 'nn'
require 'nnx'
local pl = require('pl.import_into')()

local opt = pl.lapp[[
   --dryrun  (default 10) number of iterations of a dry run not counted towards final timing
   --nGPU (default 0) number of GPUs to run on
   --batchSize (default 32) batch size
   --steps (default 1) number of steps to average performance
   --useOptnet (default true) whether to use optnet package for memory optimization
]]

local nGPU = opt.nGPU

local function calculateInputSizes(sizes)
    sizes = torch.floor((sizes - 11) / 3 + 1) -- conv1
    return sizes
end

deepSpeech = require 'cudnn_deepspeech2'


local steps = opt.steps -- nb of steps in loop to average perf
local nDryRuns = opt.dryrun
local batchSize = opt.batchSize
criterion = nn.CTCCriterion(true)
local dataset = nn.DeepSpeechDataset(batchSize)
local model = deepSpeech(batchSize, dataset.freqBins, nGPU, opt.useOptnet)
  
local inputs = torch.Tensor()
local sizes, input, targets = dataset:nextTorchSet()
input=input:view(opt.batchSize,1,dataset.freqBins, -1)

model = model
inputs:resize(input:size()):copy(input)

print('ModelType: deepspeech')

for i = 1, nDryRuns do
    model:zeroGradParameters()
    local output = model:updateOutput(inputs)
    local gradInput = model:updateGradInput(inputs, output)
    model:accGradParameters(inputs, output)
end

local tmfAvg, tmbiAvg, tRoundTripAvg = 0,0,0,0

local ok = 1
for t = 1, steps do
    local tmf, tmbi, tRoundTrip = 0, 0, 0, 0
    local roundTripTimer = torch.Timer()
    dataset = nn.DeepSpeechDataset(batchSize)
    local numberOfIterations = 0
    local sizes, input, targets = dataset:nextTorchSet()
    while (sizes ~= nil) do
        input=input:view(opt.batchSize,1,dataset.freqBins, -1)
        inputs:resize(input:size()):copy(input)        
        sys.tic()
        -- Forward through model and then criterion.
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets, calculateInputSizes(sizes))
        tmf = tmf + sys.toc()

        -- Backwards (updateGradInput, accGradParameters) including the criterion.
        sys.tic()
        model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        tmbi = tmbi + sys.toc()
--        collectgarbage()
        sizes, input, targets = dataset:nextTorchSet()
        numberOfIterations = numberOfIterations + 1
        xlua.progress(numberOfIterations * batchSize, dataset.size)
    end
    -- Divide the times to work out average time for updateOutput/updateGrad/accGrad
    tmfAvg = tmfAvg + tmf / numberOfIterations
    tmbiAvg = tmbiAvg + tmbi / numberOfIterations
    -- Add time taken for round trip of 1 epoch
    tRoundTripAvg = tRoundTripAvg + roundTripTimer:time().real
end
local tmf = tmfAvg / steps
local tmbi = tmbiAvg / steps
local tRoundTrip = tRoundTripAvg / steps
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':forward (ms):', tmf * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':backward (ms):', tmbi * 1000))

print(string.format("%-30s %25s %10.2f", 'cuDNN', ':TOTAL (ms):', (tmf + tmbi) * 1000))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples processed:', dataset.size))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Samples per second:', dataset.size / tRoundTrip))
print(string.format("%-30s %25s %10.2f", 'cuDNN', ':Seconds of audio processed per second:',  dataset.duration / tRoundTrip))

print(string.format("%-30s %25s %10.2f", 'cuDNN', ':EPOCH TIME (s):', tRoundTrip))
print()

print('')
