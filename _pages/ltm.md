---
layout: my_distill
title: "Latent Thought Models"
pretty_title: "Latent Thought Models <br><span style='font-size: 0.6em; font-weight: bold; display: block; margin-top: 0.2em;'>with Variational Bayes Inference-Time Computation</span>"
full_title: "Latent Thought Models with Variational Bayes Inference-Time Computation"
permalink: /ltm/
description: We propose Latent Thought Models (LTMs), a novel class of language models that incorporate explicit latent thought vectors to guide autoregressive generation. Through dual-rate optimization and inference-time computation, LTMs achieve superior efficiency and emergent reasoning capabilities at significantly smaller scales than traditional models.
date: 2025-06-03
future: true
htmlwidgets: true
hidden: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Deqian Kong
    url: "https://sites.google.com/view/deqiankong/"
    affiliations:
      name: UCLA
  - name: Minglu Zhao
    url: "https://mingluzhao.github.io/"
    affiliations:
      name: UCLA
  - name: Dehong Xu
    url: "https://dehongxu.github.io/"
    affiliations:
      name: UCLA
  - name: Bo Pang
    affiliations:
      name: Salesforce Research
  - name: Shu Wang
    affiliations:
      name: UCLA    
  - name: Edouardo Honig
    affiliations:
      name: UCLA 
  - name: Zhangzhang Si
    affiliations:
      name: KUNGFU.AI
  - name: Chuan Li
    affiliations:
      name: Lambda, Inc 
  - name: Jianwen Xie
    affiliations:
      name: Lambda, Inc 
  - name: Sirui Xie
    affiliations:
      name: UCLA      
  - name: Ying Nian Wu
    url: "http://www.stat.ucla.edu/~ywu/"
    affiliations:
      name: UCLA   

# must be the exact same name as your blogpost
bibliography: ltm.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: The Problem with Current AI
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## The Problem with Current AI

Think about how you write. Before putting pen to paper (or fingers to keyboard), your mind forms an abstract understanding of what you want to express. You might think about the main themes, the emotional tone, or the logical structure. Only then do you translate these abstract thoughts into concrete words.

Current language models like GPT work differently. They generate text token by token, word by word, without any higher-level planning or abstract representation. It's like speaking without thinking—impressive, but ultimately limited.

<div class="key-insight">
<strong>Key Insight:</strong> Traditional language models lack explicit mechanisms for abstract reasoning and planning, limiting their ability to perform complex cognitive tasks efficiently.
</div>

## Our Solution: Latent Thought Models

We propose **Latent Thought Models (LTMs)**—a new class of language models that explicitly learn to form abstract "thoughts" before generating text. These thoughts are represented as latent vectors that capture the essence of what the model wants to express, before translating it into actual words.

### Core Architecture

Imagine an AI that works more like a human writer:

1. **Think first**: Form abstract thoughts about the content, themes, and structure
2. **Write second**: Use these thoughts to guide the actual text generation

Our LTMs do exactly this through a two-stage process:

$$\text{Abstract Thoughts } (\mathbf{z}) \rightarrow \text{Concrete Words } (\mathbf{x})$$

The latent thought vectors $\mathbf{z}$ serve as a compressed, abstract representation that guides the generation of each token in the sequence.

{% include figure.html path="assets/img/ltm/ltm_architecture.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
    <b>LTM Architecture:</b> Latent thought vectors $\mathbf{z}$ are sampled from a prior distribution and guide autoregressive generation through cross-attention at each layer.
</div>

#### Layered Thought Vectors

Instead of having just one set of thoughts, our models use **layered thought vectors**—different abstract representations for different layers of the neural network. This creates a hierarchy of abstraction:

- **Lower layers**: Basic linguistic patterns and syntax
- **Middle layers**: Semantic relationships and concepts  
- **Higher layers**: High-level themes and narrative structure

We assume $\mathbf{z} = (\mathbf{z}_1, ..., \mathbf{z}_L)$, where $\mathbf{z}_l$ consists of thought vectors cross-attending to layer $l$ of the Transformer decoder.

#### Thought-Guided Generation

The key component is a thought-conditioned autoregressive generator $p_{\beta}(\mathbf{x}|\mathbf{z})$:

<div class="equation-highlight">
$$p_{\beta}(\mathbf{x}|\mathbf{z}) = \prod_{n=1}^N p_{\beta}(x^{(n)}|\mathbf{z}, \mathbf{x}^{(<n)})$$
</div>

Unlike standard autoregressive models that only condition on previous tokens, our model incorporates the thought vectors $\mathbf{z}$ at each generation step through cross-attention.

### Dual-Rate Learning Algorithm

Our training process mirrors human learning with a **dual-rate optimization**:
- **Fast learning**: Quick adaptation to specific examples (like episodic memory)
- **Slow learning**: Gradual accumulation of general knowledge (like procedural memory)

<div class="algorithm-box">
<strong>Algorithm: Fast-Slow Learning of LTMs</strong>

For each training batch:
1. **Fast Learning (Inference-time computation)**:
   - Initialize variational parameters $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i)$ for each sequence
   - For $t = 1$ to $T_{\text{fast}}$ steps:
     - Sample $\mathbf{z} \sim q_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i}(\mathbf{z}|\mathbf{x}_i)$
     - Compute ELBO: $\mathcal{L}_i = \mathbb{E}_{q}[\log p_{\beta}(\mathbf{x}_i|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x}_i) || p(\mathbf{z}))$
     - Update $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i)$ with high learning rate (0.3)

2. **Slow Learning**:
   - Update global decoder parameters $\beta$ with low learning rate (0.0004)
</div>

This reflects the **declarative-procedural framework** from cognitive science—our latent thoughts act like declarative memory while the text generator represents procedural knowledge.

## Demo
### Demo 1
{% raw %}
<style>
.ltm-demo-container {
    max-width: 950px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
}

.ltm-prompt-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.ltm-prompt-label {
    font-size: 0.9em;
    opacity: 0.9;
    margin-bottom: 8px;
    font-weight: 600;
}

.ltm-prompt-text {
    font-size: 1.1em;
    font-style: italic;
    font-weight: 500;
}

.ltm-layer-controls {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.ltm-layer-btn {
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    border: 2px solid #dee2e6;
    color: #495057;
    padding: 15px 20px;
    border-radius: 12px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9em;
    transition: all 0.3s ease;
    min-width: 110px;
    text-align: center;
}

.ltm-layer-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    border-color: #6c757d;
}

.ltm-layer-btn.active {
    background: linear-gradient(145deg, #4CAF50, #45a049);
    color: white;
    border-color: #4CAF50;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.ltm-layer-title {
    font-size: 1em;
    margin-bottom: 4px;
}

.ltm-layer-desc {
    font-size: 0.75em;
    opacity: 0.8;
    font-weight: 400;
}

.ltm-active-layers-display {
    text-align: center;
    margin-bottom: 25px;
    padding: 15px;
    background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
    border-radius: 10px;
    border: 2px solid #c8e6c9;
    transition: all 0.3s ease;
}

.ltm-active-layers-label {
    font-size: 0.9em;
    color: #2e7d32;
    font-weight: 600;
    margin-bottom: 8px;
}

.ltm-active-layers-text {
    font-size: 1.1em;
    color: #1b5e20;
    font-weight: 700;
}

.ltm-generation-display {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 25px;
    min-height: 120px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: all 0.4s ease;
}

.ltm-generation-text {
    font-size: 1.05em;
    line-height: 1.7;
    color: #2c3e50;
    margin-bottom: 15px;
}

.ltm-analysis-box {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-left: 4px solid #ff8a65;
    padding: 15px 20px;
    border-radius: 8px;
    margin-top: 15px;
}

.ltm-analysis-label {
    font-weight: 700;
    color: #d84315;
    margin-bottom: 8px;
}

.ltm-analysis-text {
    color: #5d4037;
    font-size: 0.95em;
    line-height: 1.5;
}

.ltm-reset-btn {
    background: linear-gradient(145deg, #ff6b6b, #ee5a52);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9em;
    margin: 0 auto;
    display: block;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.ltm-reset-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}

.ltm-instruction-box {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border: 2px solid #ff9800;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 25px;
    text-align: center;
}

.ltm-instruction-text {
    color: #e65100;
    font-weight: 600;
    margin: 0;
}

.ltm-capability-legend {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 2px solid #2196f3;
    border-radius: 12px;
    padding: 20px;
    margin-top: 30px;
}

.ltm-legend-title {
    color: #1976d2;
    font-weight: 700;
    margin-bottom: 15px;
    font-size: 1.1em;
}

.ltm-legend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.ltm-legend-item {
    padding: 12px;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 8px;
    border-left: 4px solid #2196f3;
}

.ltm-legend-layer {
    font-weight: 700;
    color: #1976d2;
    margin-bottom: 5px;
}

.ltm-legend-desc {
    font-size: 0.9em;
    color: #1565c0;
}

.ltm-fade-in {
    animation: ltmFadeIn 0.5s ease-in;
}

@keyframes ltmFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .ltm-demo-container {
        padding: 15px;
    }
    
    .ltm-layer-controls {
        gap: 8px;
    }
    
    .ltm-layer-btn {
        min-width: 90px;
        padding: 12px 15px;
        font-size: 0.8em;
    }
    
    .ltm-legend-grid {
        grid-template-columns: 1fr;
    }
}
</style>
<div class="ltm-demo-container">
    <div class="ltm-prompt-box">
        <div class="ltm-prompt-label">INPUT PROMPT:</div>
        <div class="ltm-prompt-text">"The future of artificial intelligence will fundamentally change how we..."</div>
    </div>
<div class="ltm-instruction-box">
    <p class="ltm-instruction-text">💡 Click any combination of layer groups to see how they work together!</p>
</div>

<div class="ltm-layer-controls">
    <button class="ltm-layer-btn" data-layers="1-3">
        <div class="ltm-layer-title">Layers 1-3</div>
        <div class="ltm-layer-desc">Basic Patterns</div>
    </button>
    <button class="ltm-layer-btn" data-layers="4-6">
        <div class="ltm-layer-title">Layers 4-6</div>
        <div class="ltm-layer-desc">Semantic Relations</div>
    </button>
    <button class="ltm-layer-btn" data-layers="7-9">
        <div class="ltm-layer-title">Layers 7-9</div>
        <div class="ltm-layer-desc">Abstract Reasoning</div>
    </button>
    <button class="ltm-layer-btn" data-layers="10-12">
        <div class="ltm-layer-title">Layers 10-12</div>
        <div class="ltm-layer-desc">Synthesis</div>
    </button>
</div>

<button class="ltm-reset-btn" id="ltmResetBtn">🔄 Reset All Layers</button>

<div class="ltm-active-layers-display" id="ltmActiveLayersDisplay">
    <div class="ltm-active-layers-label">ACTIVE LAYER GROUPS:</div>
    <div class="ltm-active-layers-text" id="ltmActiveLayersText">None selected</div>
</div>

<div class="ltm-generation-display" id="ltmGenerationDisplay">
    <div style="text-align: center; color: #7f8c8d; font-style: italic; margin-top: 40px;">
        Select layer groups above to see the generated text
    </div>
</div>

</div>
<script>
// Placeholder data structure - replace with your actual generations
const ltmGenerations = {
    '': {
        text: '',
        analysis: ''
    },
    '1-3': {
        text: '[PLACEHOLDER: Generation with only layers 1-3 active - basic patterns and syntax]',
        analysis: 'Only basic linguistic patterns are active. Simple word associations and grammatical structure.'
    },
    '4-6': {
        text: '[PLACEHOLDER: Generation with only layers 4-6 active - semantic understanding without syntax foundation]',
        analysis: 'Semantic relationships without proper syntactic foundation. May have meaning but poor structure.'
    },
    '7-9': {
        text: '[PLACEHOLDER: Generation with only layers 7-9 active - abstract concepts without lower-level support]',
        analysis: 'Abstract reasoning without lower-level linguistic support. High-level concepts but poor execution.'
    },
    '10-12': {
        text: '[PLACEHOLDER: Generation with only layers 10-12 active - synthesis without foundation]',
        analysis: 'Attempted synthesis without proper foundation from lower layers. May be incoherent.'
    },
    '1-3,4-6': {
        text: '[PLACEHOLDER: Generation with layers 1-6 active - syntax + semantics]',
        analysis: 'Combination of syntactic structure and semantic understanding. Coherent but may lack depth.'
    },
    '1-3,7-9': {
        text: '[PLACEHOLDER: Generation with layers 1-3 and 7-9 active - syntax + abstract reasoning]',
        analysis: 'Basic structure with abstract concepts. May have gaps in semantic connectivity.'
    },
    '1-3,10-12': {
        text: '[PLACEHOLDER: Generation with layers 1-3 and 10-12 active - syntax + synthesis]',
        analysis: 'Basic patterns with high-level synthesis. May lack semantic coherence.'
    },
    '4-6,7-9': {
        text: '[PLACEHOLDER: Generation with layers 4-9 active - semantics + abstract reasoning]',
        analysis: 'Rich semantic understanding with abstract reasoning, but may lack syntactic polish.'
    },
    '4-6,10-12': {
        text: '[PLACEHOLDER: Generation with layers 4-6 and 10-12 active - semantics + synthesis]',
        analysis: 'Semantic understanding with synthesis attempts, but may lack proper structure.'
    },
    '7-9,10-12': {
        text: '[PLACEHOLDER: Generation with layers 7-12 active - abstract reasoning + synthesis]',
        analysis: 'High-level reasoning and synthesis without lower-level linguistic foundation.'
    },
    '1-3,4-6,7-9': {
        text: '[PLACEHOLDER: Generation with layers 1-9 active - missing synthesis layer]',
        analysis: 'Strong foundation with syntax, semantics, and reasoning, but lacking final synthesis polish.'
    },
    '1-3,4-6,10-12': {
        text: '[PLACEHOLDER: Generation with layers 1-6 and 10-12 active - missing abstract reasoning]',
        analysis: 'Good structure and semantics with synthesis, but may lack deep conceptual connections.'
    },
    '1-3,7-9,10-12': {
        text: '[PLACEHOLDER: Generation with layers 1-3 and 7-12 active - missing semantic layer]',
        analysis: 'Basic structure with high-level reasoning and synthesis, but semantic gaps may exist.'
    },
    '4-6,7-9,10-12': {
        text: '[PLACEHOLDER: Generation with layers 4-12 active - missing syntax foundation]',
        analysis: 'Rich semantics, reasoning, and synthesis, but may have structural/grammatical issues.'
    },
    '1-3,4-6,7-9,10-12': {
        text: '[PLACEHOLDER: Generation with all layers 1-12 active - full model]',
        analysis: 'Complete hierarchical processing with all capabilities: syntax, semantics, reasoning, and synthesis.'
    }
};

const ltmButtons = document.querySelectorAll('.ltm-layer-btn');
const ltmDisplay = document.getElementById('ltmGenerationDisplay');
const ltmActiveLayersDisplay = document.getElementById('ltmActiveLayersText');
const ltmResetBtn = document.getElementById('ltmResetBtn');

let ltmActiveLayers = new Set();

function updateLtmDisplay() {
    const layerKey = Array.from(ltmActiveLayers).sort().join(',');
    const generation = ltmGenerations[layerKey] || ltmGenerations[''];
    
    // Update active layers display
    const activeLayersContainer = document.getElementById('ltmActiveLayersDisplay');
    if (ltmActiveLayers.size === 0) {
        ltmActiveLayersDisplay.textContent = 'None selected';
        activeLayersContainer.style.background = 'linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)';
        activeLayersContainer.style.borderColor = '#f44336';
    } else {
        const layerList = Array.from(ltmActiveLayers).sort().join(', ');
        ltmActiveLayersDisplay.textContent = layerList;
        activeLayersContainer.style.background = 'linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%)';
        activeLayersContainer.style.borderColor = '#c8e6c9';
    }
    
    // Update generation display with fade effect
    ltmDisplay.classList.remove('ltm-fade-in');
    setTimeout(() => {
        if (generation.text) {
            ltmDisplay.innerHTML = `
                <div class="ltm-generation-text">${generation.text}</div>
                <div class="ltm-analysis-box">
                    <div class="ltm-analysis-label">Analysis:</div>
                    <div class="ltm-analysis-text">${generation.analysis}</div>
                </div>
            `;
        } else {
            ltmDisplay.innerHTML = `
                <div style="text-align: center; color: #7f8c8d; font-style: italic; margin-top: 40px;">
                    Select layer groups above to see the generated text
                </div>
            `;
        }
        ltmDisplay.classList.add('ltm-fade-in');
    }, 100);
}

// Event listeners
ltmButtons.forEach(button => {
    button.addEventListener('click', () => {
        const layers = button.dataset.layers;
        
        if (ltmActiveLayers.has(layers)) {
            // Deactivate layer
            ltmActiveLayers.delete(layers);
            button.classList.remove('active');
        } else {
            // Activate layer
            ltmActiveLayers.add(layers);
            button.classList.add('active');
        }
        
        updateLtmDisplay();
    });
});

ltmResetBtn.addEventListener('click', () => {
    ltmActiveLayers.clear();
    ltmButtons.forEach(btn => btn.classList.remove('active'));
    updateLtmDisplay();
});

// Initialize display when page loads
document.addEventListener('DOMContentLoaded', function() {
    updateLtmDisplay();
});

// Also initialize immediately in case DOMContentLoaded has already fired
updateLtmDisplay();
</script>
{% endraw %}

### Demo 2
{% raw %}
<style>
.ltm-demo-container {
    max-width: 950px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
}

.ltm-prompt-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.ltm-prompt-label {
    font-size: 0.9em;
    opacity: 0.9;
    margin-bottom: 8px;
    font-weight: 600;
}

.ltm-prompt-text {
    font-size: 1.1em;
    font-style: italic;
    font-weight: 500;
}

.ltm-slider-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #dee2e6;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.ltm-slider-title {
    font-size: 1.1em;
    font-weight: 700;
    color: #495057;
    margin-bottom: 20px;
    text-align: center;
}

.ltm-range-slider {
    position: relative;
    width: 100%;
    height: 8px;
    background: #e9ecef;
    border-radius: 4px;
    margin: 30px 0;
}

.ltm-range-track {
    position: absolute;
    height: 100%;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    border-radius: 4px;
    transition: all 0.3s ease;
}

.ltm-range-input {
    position: absolute;
    top: -8px;
    width: 100%;
    height: 24px;
    background: none;
    pointer-events: none;
    -webkit-appearance: none;
    -moz-appearance: none;
}

.ltm-range-input::-webkit-slider-thumb {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #fff;
    border: 3px solid #4CAF50;
    cursor: pointer;
    pointer-events: all;
    -webkit-appearance: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: all 0.2s ease;
}

.ltm-range-input::-webkit-slider-thumb:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
}

.ltm-range-input::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #fff;
    border: 3px solid #4CAF50;
    cursor: pointer;
    pointer-events: all;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.ltm-layer-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 15px;
    font-size: 0.9em;
    color: #6c757d;
}

.ltm-selected-range {
    text-align: center;
    margin-top: 15px;
    padding: 10px;
    background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
    border-radius: 8px;
    border: 2px solid #c8e6c9;
}

.ltm-selected-range-label {
    font-size: 0.9em;
    color: #2e7d32;
    font-weight: 600;
    margin-bottom: 5px;
}

.ltm-selected-range-text {
    font-size: 1.2em;
    color: #1b5e20;
    font-weight: 700;
}

.ltm-preset-buttons {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.ltm-preset-btn {
    background: linear-gradient(145deg, #fff, #f8f9fa);
    border: 2px solid #dee2e6;
    color: #6c757d;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.8em;
    font-weight: 600;
    transition: all 0.3s ease;
}

.ltm-preset-btn:hover {
    border-color: #4CAF50;
    color: #4CAF50;
    transform: translateY(-1px);
}

.ltm-generation-display {
    background: white;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 25px;
    margin-bottom: 25px;
    min-height: 120px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: all 0.4s ease;
}

.ltm-generation-text {
    font-size: 1.05em;
    line-height: 1.7;
    color: #2c3e50;
    margin-bottom: 15px;
}

.ltm-analysis-box {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-left: 4px solid #ff8a65;
    padding: 15px 20px;
    border-radius: 8px;
    margin-top: 15px;
}

.ltm-analysis-label {
    font-weight: 700;
    color: #d84315;
    margin-bottom: 8px;
}

.ltm-analysis-text {
    color: #5d4037;
    font-size: 0.95em;
    line-height: 1.5;
}

.ltm-capability-info {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border: 2px solid #2196f3;
    border-radius: 12px;
    padding: 20px;
    margin-top: 30px;
}

.ltm-info-title {
    color: #1976d2;
    font-weight: 700;
    margin-bottom: 15px;
    font-size: 1.1em;
}

.ltm-layer-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    font-size: 0.9em;
}

.ltm-layer-range {
    padding: 8px;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 6px;
    text-align: center;
}

.ltm-layer-range-num {
    font-weight: 700;
    color: #1976d2;
}

.ltm-layer-range-desc {
    color: #1565c0;
    font-size: 0.85em;
}

.ltm-fade-in {
    animation: ltmFadeIn 0.5s ease-in;
}

@keyframes ltmFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .ltm-demo-container {
        padding: 15px;
    }
    
    .ltm-preset-buttons {
        gap: 6px;
    }
    
    .ltm-preset-btn {
        padding: 6px 12px;
        font-size: 0.75em;
    }
}
</style>

<div class="ltm-demo-container">
    <div class="ltm-prompt-box">
        <div class="ltm-prompt-label">INPUT PROMPT:</div>
        <div class="ltm-prompt-text">"The future of artificial intelligence will fundamentally change how we..."</div>
    </div>
    
    <div class="ltm-slider-container">
        <div class="ltm-slider-title">🧠 Select Layer Range</div>
        
        <div class="ltm-range-slider" id="rangeSlider">
            <div class="ltm-range-track" id="rangeTrack"></div>
            <input type="range" min="1" max="12" value="1" class="ltm-range-input" id="rangeMin">
            <input type="range" min="1" max="12" value="12" class="ltm-range-input" id="rangeMax">
        </div>
        
        <div class="ltm-layer-labels">
            <span>Layer 1</span>
            <span>Layer 3</span>
            <span>Layer 6</span>
            <span>Layer 9</span>
            <span>Layer 12</span>
        </div>
        
        <div class="ltm-selected-range" id="selectedRange">
            <div class="ltm-selected-range-label">ACTIVE LAYERS:</div>
            <div class="ltm-selected-range-text" id="selectedRangeText">Layers 1-12</div>
        </div>
        
        <div class="ltm-preset-buttons">
            <button class="ltm-preset-btn" data-range="1,3">Layers 1-3</button>
            <button class="ltm-preset-btn" data-range="4,6">Layers 4-6</button>
            <button class="ltm-preset-btn" data-range="7,9">Layers 7-9</button>
            <button class="ltm-preset-btn" data-range="10,12">Layers 10-12</button>
            <button class="ltm-preset-btn" data-range="1,6">Layers 1-6</button>
            <button class="ltm-preset-btn" data-range="7,12">Layers 7-12</button>
            <button class="ltm-preset-btn" data-range="1,12">All Layers</button>
        </div>
    </div>
    
    <div class="ltm-generation-display" id="generationDisplay">
        <div style="text-align: center; color: #7f8c8d; font-style: italic; margin-top: 40px;">
            Adjust the layer range above to see how different layer combinations affect text generation
        </div>
    </div>
    
    <div class="ltm-capability-info">
        <div class="ltm-info-title">💡 Layer Capabilities Overview:</div>
        <div class="ltm-layer-info">
            <div class="ltm-layer-range">
                <div class="ltm-layer-range-num">Layers 1-3</div>
                <div class="ltm-layer-range-desc">Basic patterns & syntax</div>
            </div>
            <div class="ltm-layer-range">
                <div class="ltm-layer-range-num">Layers 4-6</div>
                <div class="ltm-layer-range-desc">Semantic relationships</div>
            </div>
            <div class="ltm-layer-range">
                <div class="ltm-layer-range-num">Layers 7-9</div>
                <div class="ltm-layer-range-desc">Abstract reasoning</div>
            </div>
            <div class="ltm-layer-range">
                <div class="ltm-layer-range-num">Layers 10-12</div>
                <div class="ltm-layer-range-desc">Synthesis & coherence</div>
            </div>
        </div>
    </div>
</div>

<script>
// Sample generation data - replace with your actual model outputs
const generationData = {
    generateText: function(startLayer, endLayer) {
        const layerRanges = {
            '1-3': 'Basic syntactic patterns and simple word associations. Limited semantic depth.',
            '4-6': 'Semantic understanding with conceptual connections but may lack syntactic polish.',
            '7-9': 'Abstract reasoning and thematic coherence with sophisticated conceptual integration.',
            '10-12': 'High-level synthesis with argumentative structure and technical precision.',
            '1-6': 'Foundation combining syntax and semantics. Coherent structure with meaningful content.',
            '7-12': 'Advanced reasoning and synthesis. Rich conceptual depth with polished execution.',
            '1-12': 'Complete hierarchical processing with all capabilities integrated.'
        };
        
        // Generate description based on range
        let description = '';
        let analysis = '';
        
        if (startLayer === 1 && endLayer === 12) {
            description = 'The future of artificial intelligence will fundamentally change how we understand the nature of thought itself, as models like LTMs demonstrate that explicit reasoning processes can emerge from architectural innovations rather than sheer scale. This shift toward inference-time computation and hierarchical abstraction suggests that the next generation of AI will be characterized not by larger parameters, but by more sophisticated cognitive architectures that mirror the declarative-procedural frameworks underlying human intelligence, ultimately leading to AI systems that are both more capable and more aligned with human reasoning patterns.';
            analysis = 'Complete hierarchical processing with all capabilities: syntax, semantics, reasoning, and synthesis working together seamlessly.';
        } else if (startLayer <= 3 && endLayer <= 3) {
            description = 'The future of artificial intelligence will fundamentally change how we work and live. AI systems will become more powerful and will be able to do many things that humans can do. This will have a big impact on society and the economy.';
            analysis = 'Only basic linguistic patterns are active. Simple word associations and grammatical structure without deeper understanding.';
        } else if (startLayer <= 6 && endLayer <= 6) {
            description = 'The future of artificial intelligence will fundamentally change how we approach complex problem-solving, from scientific research to creative endeavors. As AI systems develop more sophisticated reasoning capabilities, they will augment human intelligence rather than simply replace it.';
            analysis = 'Semantic understanding emerges with conceptual connections, but may lack the polish of higher-level synthesis.';
        } else if (startLayer >= 7 && endLayer <= 9) {
            description = 'The future of artificial intelligence will fundamentally change how we conceptualize intelligence itself, blurring the boundaries between human cognition and machine reasoning through emergent capabilities and paradigm shifts toward interpretable, human-aligned artificial minds.';
            analysis = 'Abstract reasoning and meta-cognitive awareness without lower-level linguistic foundation may result in conceptual richness but structural gaps.';
        } else if (startLayer >= 10) {
            description = 'The future of artificial intelligence will represent a synthesis of architectural innovations, inference-time computation, and sophisticated cognitive frameworks that characterize next-generation AI systems with enhanced capability and alignment.';
            analysis = 'High-level synthesis attempts without proper foundation from lower layers. May be conceptually advanced but lack coherent structure.';
        } else {
            // Mixed ranges
            description = `[Layer range ${startLayer}-${endLayer}]: The future of artificial intelligence will fundamentally change how we approach the intersection of computational reasoning and human-like cognitive processes, demonstrating varying degrees of linguistic sophistication, semantic understanding, and conceptual integration depending on the active layer configuration.`;
            analysis = `Layer range ${startLayer}-${endLayer} shows a combination of capabilities. Lower layers contribute structural foundation while higher layers add conceptual depth and synthesis.`;
        }
        
        return { text: description, analysis: analysis };
    }
};

const rangeMin = document.getElementById('rangeMin');
const rangeMax = document.getElementById('rangeMax');
const rangeTrack = document.getElementById('rangeTrack');
const selectedRangeText = document.getElementById('selectedRangeText');
const generationDisplay = document.getElementById('generationDisplay');
const presetButtons = document.querySelectorAll('.ltm-preset-btn');

function updateSlider() {
    const min = parseInt(rangeMin.value);
    const max = parseInt(rangeMax.value);
    
    // Ensure min doesn't exceed max
    if (min > max) {
        rangeMin.value = max;
        return updateSlider();
    }
    
    // Update visual track
    const percent1 = ((min - 1) / 11) * 100;
    const percent2 = ((max - 1) / 11) * 100;
    
    rangeTrack.style.left = percent1 + '%';
    rangeTrack.style.width = (percent2 - percent1) + '%';
    
    // Update text display
    if (min === max) {
        selectedRangeText.textContent = `Layer ${min}`;
    } else {
        selectedRangeText.textContent = `Layers ${min}-${max}`;
    }
    
    // Update generation display
    updateGeneration(min, max);
}

function updateGeneration(startLayer, endLayer) {
    const generation = generationData.generateText(startLayer, endLayer);
    
    generationDisplay.classList.remove('ltm-fade-in');
    setTimeout(() => {
        generationDisplay.innerHTML = `
            <div class="ltm-generation-text">${generation.text}</div>
            <div class="ltm-analysis-box">
                <div class="ltm-analysis-label">Analysis:</div>
                <div class="ltm-analysis-text">${generation.analysis}</div>
            </div>
        `;
        generationDisplay.classList.add('ltm-fade-in');
    }, 100);
}

// Event listeners
rangeMin.addEventListener('input', updateSlider);
rangeMax.addEventListener('input', updateSlider);

// Preset buttons
presetButtons.forEach(button => {
    button.addEventListener('click', () => {
        const range = button.dataset.range.split(',');
        rangeMin.value = range[0];
        rangeMax.value = range[1];
        updateSlider();
        
        // Visual feedback
        button.style.transform = 'scale(0.95)';
        setTimeout(() => {
            button.style.transform = 'translateY(-1px)';
        }, 150);
    });
});

// Initialize
updateSlider();
</script>
{% endraw %}

## Key Insights and Breakthroughs

### New Scaling Dimensions

While traditional language models scale along two main axes (model size and training data), LTMs introduce a third crucial dimension: **inference steps**. More thinking time leads to better performance—you can trade off model size for more deliberate reasoning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/ppl_val_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Scaling behaviors over training tokens and compute.</b> Models with more inference steps demonstrate improved sample efficiency and become compute-efficient beyond certain training compute thresholds.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/scaling_tokens_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/scaling_flops_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Scaling behaviors over training tokens and compute.</b> We plot the performance of LTM training runs ($N_{z}=24$) across inference steps ($T_\mathrm{fast}=$16-64) and model sizes (38M-76M). Models with more inference steps demonstrate improved sample efficiency and become compute-efficient beyond certain training compute thresholds.
</div>

### Emergent Few-Shot Learning at Small Scale

Something remarkable happens with LTMs: they develop few-shot learning abilities (like GPT-3's in-context learning) but with **dramatically fewer parameters**. Our smallest model achieves this with just 38M parameters—a fraction of what's typically needed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/gsm8k.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Arithmetic reasoning on GSM8K:</b> LTMs with few-shot demonstrations outperform much larger GPT-2 models across various settings.
</div>

### Superior Efficiency

The results speak for themselves:
- **76% fewer training tokens** needed compared to similar-sized models
- **93% fewer parameters** than GPT-2-Large while matching its performance
- **5× faster generation** compared to some diffusion-based alternatives

## Technical Deep Dive

### Model Formulation

We formulate LTMs within the classical variational Bayes framework. The model assumes latent thought vectors $\mathbf{z}$ follow a prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and generate text $\mathbf{x}$ via a Transformer decoder.

We introduce a sequence-specific variational posterior $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ and maximize the evidence lower bound (ELBO):

<div class="equation-highlight">
$$\mathcal{L}(\beta, \boldsymbol{\mu}, \boldsymbol{\sigma}^2) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p_{\beta}(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))$$
</div>

Crucially, $(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ are **local parameters** specific to each sequence, while $\beta$ represents **global parameters** shared across all samples.

### Inference-Time Computation

LTMs introduce a distinct computational cost: **inference-time computation** stemming from the fast learning of latent thought vectors. This occurs in both training and testing.

For a model with $L$ layers, $N_{\mathbf{z}}$ latent vectors per layer, and $T_{\text{fast}}$ inference steps, the computational complexity scales as:

$$\mathcal{O}(T_{\text{fast}} \cdot L \cdot (N^2H + NN_{\mathbf{z}}H + NH^2))$$

When $T_{\text{fast}} \gg 1$, the inference computation dominates, making thinking time the primary computational factor.

## Experimental Results

We conducted extensive experiments at GPT-2 scale using the OpenWebText dataset. Our results demonstrate:

**Zero-shot Language Modeling Performance:**

| Model | Parameters | Training FLOPs/tok | PTB | WikiText | LM1B |
|-------|------------|-------------------|-----|----------|------|
| GPT-2-Large | 762M | 5.32G | 161.33 | 30.09 | 45.61 |
| LTM-Medium | 51M | 5.52G | ≤32.06 | ≤17.39 | ≤25.16 |
| LTM-Large | 76M | 32.2G | ≤**4.43** | ≤**3.66** | ≤**3.92** |

**Text Generation Quality:**

| Model | Sampling | MAUVE ↑ |
|-------|----------|---------|
| GPT-2-Medium | Nucleus-0.95 | 0.955 |
| GPT-2-Medium | Multinomial | 0.802 |
| LTM-Large | Multinomial | **0.974** |
| LTM-Large | Greedy | 0.972 |

### Probing the Latent Thoughts

We investigated how semantic information is distributed across layers through progressive reconstruction experiments. The results reveal that LTMs process information hierarchically:

- **6-layer models**: Gradual improvement (~55% accuracy) followed by sharp synthesis in the final layer
- **12-layer models**: Distributed processing with steady increases through layers 1-8 (~65%) and crucial integration at layers 9-10 (>95% accuracy)

This demonstrates distinctive "synthesis layers" that integrate information from earlier representations.

## What This Means for AI

### The Language of Thought Hypothesis

Our approach connects to a deep idea in cognitive science: that thinking happens in an internal "language of thought" that's distinct from the language we speak. The latent thought vectors can be seen as "words" in this internal cognitive language.

### Inference-Time Computation as a New Paradigm

Perhaps most importantly, LTMs demonstrate that **thinking time** can be as valuable as model size or training data. This opens up new possibilities:

- **Adaptive computation**: Thinking harder on difficult problems
- **Resource allocation**: Trading model size for inference time based on computational budgets
- **Reasoning capabilities**: Using iterative refinement for complex problem-solving

## Looking Forward

This work opens several exciting directions:

1. **Structured Prior Models**: Moving beyond simple Gaussian priors to more sophisticated reasoning structures
2. **Reward-Guided Thinking**: Using reward models to guide the thinking process toward better outcomes  
3. **Hierarchical Abstraction**: Developing even more sophisticated multi-level thought representations

### Current Limitations

We acknowledge important areas for future work:

- **Learnable Prior Models**: Our current Gaussian prior is simple—more structured priors could enable even richer reasoning
- **Reward Models in Latent Space**: Incorporating verifier models $p_\gamma(r|\mathbf{z})$ to guide optimization for reasoning tasks

## Conclusion

Latent Thought Models represent a fundamental shift in how we think about language generation. Instead of immediate word-by-word generation, they introduce a more human-like process of abstract thinking followed by linguistic expression.

The key insight is simple but profound: **giving AI systems explicit space to think makes them more efficient, more capable, and more aligned with how humans actually process language**.

As we continue to push the boundaries of AI capabilities, approaches like LTMs suggest that the future lies not just in bigger models or more data, but in architectures that more closely mirror the sophisticated cognitive processes that make human intelligence so remarkable.


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-04-28-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):



