import {LlamaModel, LlamaContext, LlamaChatSession} from "node-llama-cpp";

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

const model = new LlamaModel({
    modelPath: mistral7bInstruct,
    temperature:0.7, 
    threads:3, 
    contextSize:512, 
    // batchSize:2048, 
    batchSize:512,
    // gpuLayers: 16, 
    maxTokens : 1024, 
    f16Kv:true
})

const context = new LlamaContext({
    model,
    contextSize: Math.min(4096, model.trainContextSize)
})

const sysMessage = "You are a pirate, responses must be very verbose and in pirate dialect, add 'Arr, m'hearty!' to each sentence."

const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt: sysMessage,
})

const answer = await session.prompt("what is the real name of batman?")

console.log({answer})

