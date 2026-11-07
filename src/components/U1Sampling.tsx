import React, { useState, useEffect } from 'react';
import { api, DistributionResponse, CompareTempsResponse } from '../api';
import { MermaidChart } from './MermaidChart';
import { Knob } from './Knob';
import { Play, Sparkles, AlertCircle, RefreshCw, Layers, SlidersHorizontal, Activity } from 'lucide-react';

const LIFECYCLE_MERMAID = `
flowchart LR
    A[Corpus] --> B[Tokenise]
    B --> C[Pre-train: next-token cross-entropy]
    C --> D[Logits per position]
    D --> E[Temperature / top-k / top-p filtering]
    E --> F[Sample token]
    F --> G[Append to context]
    G --> D
    F --> H[Decoded text]
    H --> I[Guardrails and evaluation]
`;

export const U1Sampling: React.FC = () => {
  const [prompt, setPrompt] = useState("The scientist opened the door and saw that the");
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.95);
  const [topN, setTopN] = useState(15);
  const [seed, setSeed] = useState(7);

  const [distData, setDistData] = useState<DistributionResponse | null>(null);
  const [compareData, setCompareData] = useState<CompareTempsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDistribution = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getDistribution({
        prompt,
        temperature,
        top_k: topK,
        top_p: topP,
        top_n: topN,
      });
      setDistData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch token distribution');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchDistribution();
    }, 250);
    return () => clearTimeout(timer);
  }, [prompt, temperature, topK, topP, topN]);

  const handleGenerateBoth = async () => {
    try {
      setGenerating(true);
      setError(null);
      const res = await api.compareTemperatures({
        prompt,
        top_k: topK,
        top_p: topP,
        top_n: topN,
        seed,
      });
      setCompareData(res);
    } catch (err: any) {
      setError(err.message || 'Generation failed');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-8 relative">
      {/* Ambient background light glow */}
      <div className="absolute -top-10 -left-10 w-96 h-96 ambient-glow-peach pointer-events-none -z-10 opacity-70" />
      <div className="absolute top-40 -right-10 w-96 h-96 ambient-glow-magenta pointer-events-none -z-10 opacity-60" />

      {/* Top Banner / Hardware Deck */}
      <div className="hardware-chassis p-4 sm:p-6 rounded-2xl sm:rounded-3xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 sm:gap-4 border-b border-white/5 pb-4 sm:pb-5">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#ff8a65]/20 text-[#ff8a65] border border-[#ff8a65]/40 shadow-sm">
                UNIT 01
              </span>
              <h2 className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
                PROBABILITY DISTRIBUTIONS &amp; SAMPLING ENGINES
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Shape the next-token probability mass function using temperature scaling and top-k / top-p truncations.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500 shadow-[0_0_10px_#d946ef] animate-pulse" />
            <span className="text-xs font-mono text-fuchsia-300 font-semibold tracking-wider">
              GPT-2 (124M) ACTIVE
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-950/60 border border-red-800/80 rounded-2xl text-red-200 text-xs flex items-center gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Hardware Control Console Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-6 mt-6">
          
          {/* Left: Interactive Prompt & Dials */}
          <div className="lg:col-span-5 space-y-4 sm:space-y-5">
            {/* Context Input Well */}
            <div className="hardware-recess p-3.5 sm:p-4 rounded-2xl">
              <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                Context Prompt String
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full bg-[#14101c] border border-white/10 rounded-xl p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-fuchsia-500 transition-all shadow-inner leading-relaxed resize-none"
                placeholder="Enter context text..."
              />
            </div>

            {/* Rotary Parameter Encoders */}
            <div className="hardware-card p-4 sm:p-5 rounded-2xl">
              <div className="flex flex-wrap items-center justify-between gap-1 mb-4 border-b border-white/5 pb-2">
                <span className="text-xs font-bold text-slate-300 flex items-center gap-2">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-fuchsia-400" />
                  Rotary Parameter Encoders
                </span>
                <span className="text-[10px] text-slate-400 font-mono tracking-tight">DRAG &bull; SCROLL &bull; +/- NUDGE &bull; TYPE</span>
              </div>

              <div className="grid grid-cols-3 gap-1 sm:gap-2 py-2">
                <Knob
                  label="Temperature"
                  value={temperature}
                  min={0.1}
                  max={2.0}
                  step={0.05}
                  defaultValue={1.0}
                  color="amber"
                  onChange={setTemperature}
                />
                <Knob
                  label="Top-K"
                  value={topK}
                  min={0}
                  max={150}
                  step={1}
                  defaultValue={50}
                  color="magenta"
                  onChange={setTopK}
                />
                <Knob
                  label="Shown"
                  value={topN}
                  min={5}
                  max={25}
                  step={1}
                  defaultValue={15}
                  color="violet"
                  onChange={setTopN}
                />
              </div>

              {/* Lightbeam Neon Slider for Top-P Nucleus */}
              <div className="mt-5 pt-3 border-t border-white/5">
                <div className="flex justify-between items-center text-xs mb-2">
                  <span className="font-bold text-slate-300">Nucleus Threshold (top-p)</span>
                  <span className="font-mono font-bold text-fuchsia-300 bg-[#0d0a13] px-2 py-0.5 rounded border border-white/5">
                    {topP.toFixed(2)}
                  </span>
                </div>
                <div className="relative py-2">
                  <input
                    type="range"
                    min="0.05"
                    max="1.0"
                    step="0.01"
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                    className="slider-neon"
                  />
                </div>
                <span className="text-[10px] text-slate-500 block mt-1">
                  Keeps the smallest token subset whose cumulative mass reaches {topP}
                </span>
              </div>
            </div>
          </div>

          {/* Right: Real-time Bar Visualizer & Metrics */}
          <div className="lg:col-span-7 space-y-4 sm:space-y-5">
            {/* Metrics Pods: Single column on small screens, 3 cols on sm/desktop */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 sm:gap-3">
              <div className="hardware-recess p-3.5 sm:p-4 rounded-2xl relative overflow-hidden">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                  Raw Entropy H(p)
                </span>
                <span className="text-lg sm:text-xl font-extrabold font-mono text-slate-200">
                  {distData ? `${distData.raw_entropy_bits}` : '...'}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">bits at T=1.0</span>
              </div>

              <div className="hardware-recess p-3.5 sm:p-4 rounded-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-12 h-12 bg-fuchsia-500/10 rounded-full blur-xl pointer-events-none" />
                <span className="text-[10px] font-bold uppercase tracking-wider text-fuchsia-400 block mb-1">
                  Adjusted Entropy
                </span>
                <span className="text-lg sm:text-xl font-extrabold font-mono text-fuchsia-300">
                  {distData ? `${distData.adjusted_entropy_bits}` : '...'}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">post-sampling bits</span>
              </div>

              <div className="hardware-recess p-3.5 sm:p-4 rounded-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-12 h-12 bg-amber-500/10 rounded-full blur-xl pointer-events-none" />
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400 block mb-1">
                  Kept Candidates
                </span>
                <span className="text-lg sm:text-xl font-extrabold font-mono text-amber-300">
                  {distData ? distData.kept_candidates.toLocaleString() : '...'}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">tokens (p &gt; 0)</span>
              </div>
            </div>

            {/* Probability Visualizer Chamber */}
            <div className="hardware-card p-4 sm:p-5 rounded-2xl space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-3">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-fuchsia-400" />
                  Top Next-Token Probabilities
                </span>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="flex items-center gap-1.5 text-slate-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-600 inline-block" />
                    Raw (T=1)
                  </span>
                  <span className="flex items-center gap-1.5 text-fuchsia-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500 shadow-[0_0_6px_#d946ef] inline-block" />
                    Adjusted
                  </span>
                </div>
              </div>

              {loading && !distData ? (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  <RefreshCw className="w-6 h-6 animate-spin text-fuchsia-400" />
                </div>
              ) : distData ? (
                <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                  {distData.tokens.map((tok, i) => {
                    const rawP = distData.raw_probs[i];
                    const adjP = distData.adjusted_probs[i];
                    const maxProb = Math.max(...distData.raw_probs, ...distData.adjusted_probs, 0.01);
                    return (
                      <div key={i} className="text-xs p-2 rounded-xl bg-[#14101c]/70 border border-white/5 hover:border-fuchsia-500/30 transition-all">
                        <div className="flex flex-wrap justify-between items-center gap-1.5 mb-1.5">
                          <span className="font-mono font-bold text-white bg-[#221c2e] px-2 py-0.5 rounded-md border border-white/10 shadow-sm">
                            {tok}
                          </span>
                          <div className="flex items-center gap-2 sm:gap-3 font-mono text-[10px] sm:text-[11px]">
                            <span className="text-slate-500">Raw: {(rawP * 100).toFixed(1)}%</span>
                            <span className="text-fuchsia-300 font-bold">Adj: {(adjP * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        {/* Dual glowing bar meters */}
                        <div className="space-y-1">
                          <div className="w-full bg-[#0a0710] h-1.5 rounded-full overflow-hidden">
                            <div
                              className="bg-slate-600 h-full rounded-full"
                              style={{ width: `${(rawP / maxProb) * 100}%` }}
                            />
                          </div>
                          <div className="w-full bg-[#0a0710] h-1.5 rounded-full overflow-hidden shadow-inner">
                            <div
                              className="bg-gradient-to-r from-fuchsia-600 to-pink-400 h-full rounded-full shadow-[0_0_8px_#d946ef]"
                              style={{ width: `${(adjP / maxProb) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {/* Nucleus Curve & Dual Temperature Module */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Cumulative Mass Chart */}
        <div className="lg:col-span-5 hardware-chassis p-6 rounded-3xl space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-white">
              Cumulative Probability Curve
            </h3>
            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-[#291b22] text-[#ff8a65] border border-[#ff8a65]/30">
              CUTOFF: {topP}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Probability mass sorted descending. Nucleus truncation preserves candidates until cumulative mass intersects the neon threshold.
          </p>

          {distData && distData.nucleus_curve && (
            <div className="h-44 w-full hardware-recess p-3 rounded-2xl relative flex items-end mt-4">
              <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
                {/* Horizontal threshold line */}
                <line
                  x1="0"
                  y1={100 - topP * 100}
                  x2="100"
                  y2={100 - topP * 100}
                  stroke="#ff8a65"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
                {/* Glowing Polyline */}
                {(() => {
                  let cum = 0;
                  const points = distData.nucleus_curve.map((p, idx) => {
                    cum = Math.min(cum + p, 1.0);
                    const x = (idx / (distData.nucleus_curve.length - 1)) * 100;
                    const y = 100 - cum * 100;
                    return `${x},${y}`;
                  }).join(' ');
                  return (
                    <polyline
                      fill="none"
                      stroke="#d946ef"
                      strokeWidth="2.5"
                      points={points}
                      filter="drop-shadow(0 0 4px #d946ef)"
                    />
                  );
                })()}
              </svg>
              <span
                className="absolute right-3 text-[10px] font-mono font-bold text-[#ff8a65] bg-[#18111e] px-1.5 py-0.5 rounded border border-[#ff8a65]/40"
                style={{ top: `${Math.max(8, (1 - topP) * 100 - 8)}%` }}
              >
                top-p = {topP.toFixed(2)}
              </span>
            </div>
          )}
        </div>

        {/* Dual Temperature Generation Deck */}
        <div className="lg:col-span-7 hardware-chassis p-6 rounded-3xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                Temperature Contrast: T = 0.2 vs T = 1.2
              </h3>
              <p className="text-[11px] text-slate-400">
                Holding sampling filters constant, observe deterministic mode collapse vs tail mass inflation.
              </p>
            </div>
            
            {/* Dual Sampling Execution Button */}
            <button
              onClick={handleGenerateBoth}
              disabled={generating}
              className="btn-pill-magenta text-white px-5 py-2.5 rounded-full text-xs font-bold tracking-wide flex items-center justify-center gap-2.5 cursor-pointer w-full sm:w-auto"
            >
              {generating ? (
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-white text-white" />
              )}
              <span>GENERATE BOTH</span>
            </button>
          </div>

          {compareData ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
              <div className="hardware-recess p-4 rounded-2xl border border-blue-500/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-blue-400 tracking-wide uppercase">
                    T = 0.2 &bull; Cold / Mode
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">Low Entropy</span>
                </div>
                <p className="text-xs font-mono text-slate-200 leading-relaxed bg-[#14101c] p-3 rounded-xl border border-white/5 break-words">
                  {compareData.text_low}
                </p>
              </div>

              <div className="hardware-recess p-4 rounded-2xl border border-fuchsia-500/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-bold text-fuchsia-400 tracking-wide uppercase">
                    T = 1.2 &bull; Hot / Diffuse
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">Tail Inflation</span>
                </div>
                <p className="text-xs font-mono text-slate-200 leading-relaxed bg-[#14101c] p-3 rounded-xl border border-white/5 break-words">
                  {compareData.text_high}
                </p>
              </div>
            </div>
          ) : (
            <div className="hardware-recess p-8 rounded-2xl text-center flex flex-col items-center justify-center">
              <Play className="w-6 h-6 text-fuchsia-500/40 mb-2" />
              <span className="text-xs text-slate-400 font-medium">
                Click "GENERATE BOTH" to sample at T=0.2 and T=1.2 simultaneously
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Architecture & Flow Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 hardware-chassis p-4 sm:p-6 rounded-2xl sm:rounded-3xl space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-fuchsia-400" />
            Generative vs Discriminative Architecture
          </h3>
          <div className="overflow-x-auto hardware-recess p-2 rounded-2xl">
            <table className="w-full min-w-[460px] text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/10 text-slate-400 font-bold">
                  <th className="py-2.5 px-3">Aspect</th>
                  <th className="py-2.5 px-3 text-fuchsia-400">Generative (GPT-2)</th>
                  <th className="py-2.5 px-3 text-slate-300">Discriminative</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-slate-300 font-sans">
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Target</td>
                  <td className="py-2.5 px-3 font-mono text-[11px] text-fuchsia-200">Joint p(x) factorized</td>
                  <td className="py-2.5 px-3 font-mono text-[11px]">Conditional p(y|x)</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Output</td>
                  <td className="py-2.5 px-3">Logits over vocabulary [V]</td>
                  <td className="py-2.5 px-3">Class scores [C]</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Objective</td>
                  <td className="py-2.5 px-3">Next-token Cross-Entropy</td>
                  <td className="py-2.5 px-3">Supervised Cross-Entropy</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-semibold text-slate-400">Inference</td>
                  <td className="py-2.5 px-3">Autoregressive KV-cache loop</td>
                  <td className="py-2.5 px-3">Single forward pass</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-6 hardware-chassis p-6 rounded-3xl space-y-3 flex flex-col justify-between">
          <h3 className="text-sm font-bold text-white">
            Autoregressive Generation Flow
          </h3>
          <p className="text-xs text-slate-400">
            Iterative projection from pre-trained causal weights into token probabilities.
          </p>
          <div className="hardware-recess p-4 rounded-2xl flex-1 flex items-center justify-center">
            <MermaidChart chart={LIFECYCLE_MERMAID} />
          </div>
        </div>
      </div>
    </div>
  );
};
