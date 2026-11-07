import React, { useState } from 'react';
import { api, GenerateResponse, ImageSynthResponse } from '../api';
import { Knob } from './Knob';
import { Play, Image as ImageIcon, AlertCircle, RefreshCw, Cpu, Sparkles } from 'lucide-react';

export const U4SamplerImage: React.FC = () => {
  const [prompt, setPrompt] = useState("A quiet harbour at dawn, with");
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.95);
  const [tokens, setTokens] = useState(25);
  const [seed, setSeed] = useState(0);

  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);
  const [imageData, setImageData] = useState<ImageSynthResponse | null>(null);
  const [sampling, setSampling] = useState(false);
  const [imaging, setImaging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSample = async () => {
    try {
      setSampling(true);
      setError(null);
      setImageData(null);
      const res = await api.generateWithTrace({
        prompt,
        max_new_tokens: tokens,
        temperature,
        top_k: topK,
        top_p: topP,
        seed,
      });
      setGenResult(res);
    } catch (err: any) {
      setError(err.message || 'Sampling loop failed');
    } finally {
      setSampling(false);
    }
  };

  const handleSynthesizeImage = async () => {
    if (!genResult) return;
    try {
      setImaging(true);
      setError(null);
      const res = await api.synthesizeImage({
        text: genResult.text,
        seed,
      });
      setImageData(res);
    } catch (err: any) {
      setError(err.message || 'Image synthesis failed');
    } finally {
      setImaging(false);
    }
  };

  return (
    <div className="space-y-8 relative">
      {/* Ambient background light glow */}
      <div className="absolute top-20 -left-10 w-96 h-96 ambient-glow-peach pointer-events-none -z-10 opacity-50" />
      <div className="absolute top-80 -right-10 w-96 h-96 ambient-glow-magenta pointer-events-none -z-10 opacity-50" />

      {/* Header */}
      <div className="hardware-chassis p-6 rounded-3xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#a855f7]/20 text-[#c084fc] border border-[#a855f7]/40 shadow-sm">
                UNIT 04
              </span>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                FROM-SCRATCH PYTORCH SAMPLER &amp; SD-TURBO SYNTHESIS
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Step through the autoregressive KV-cache loop in pure PyTorch, record per-step candidate entropy, and synthesize matching imagery with sd-turbo.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-pink-500 shadow-[0_0_10px_#ec4899] animate-pulse" />
            <span className="text-xs font-mono text-pink-300 font-semibold tracking-wider">
              KV-CACHE + DIFFUSERS
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-950/60 border border-red-800/80 rounded-2xl text-red-200 text-xs flex items-center gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Sampler Controls Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
          
          {/* Controls Deck */}
          <div className="lg:col-span-5 space-y-5">
            <div className="hardware-recess p-4 rounded-2xl">
              <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                Seed Prompt String
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full bg-[#14101c] border border-white/10 rounded-xl p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-fuchsia-500 transition-all resize-none shadow-inner leading-relaxed"
              />
            </div>

            {/* Knobs */}
            <div className="hardware-card p-5 rounded-2xl space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <span className="text-xs font-bold text-slate-300">Sampling Knobs</span>
                <span className="text-[10px] text-slate-400 font-mono tracking-tight">DRAG &bull; SCROLL &bull; +/- NUDGE &bull; TYPE</span>
              </div>

              <div className="grid grid-cols-3 gap-2 py-2">
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
                  max={100}
                  step={1}
                  defaultValue={50}
                  color="magenta"
                  onChange={setTopK}
                />
                <Knob
                  label="Tokens"
                  value={tokens}
                  min={5}
                  max={60}
                  step={1}
                  defaultValue={25}
                  color="violet"
                  onChange={setTokens}
                />
              </div>

              {/* Lightbeam Slider for Top-p */}
              <div className="pt-2">
                <div className="flex justify-between items-center text-xs mb-1.5">
                  <span className="font-bold text-slate-300">Nucleus (top-p)</span>
                  <span className="font-mono text-fuchsia-300 font-bold bg-[#0d0a13] px-2 py-0.5 rounded border border-white/5">
                    {topP.toFixed(2)}
                  </span>
                </div>
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

              <div className="pt-2 flex items-center justify-between">
                <span className="text-xs text-slate-400 font-medium">Random Generator Seed:</span>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                  className="w-24 bg-[#0e0b14] border border-white/10 rounded-xl px-2 py-1 text-xs font-mono text-center text-slate-200"
                />
              </div>

              {/* Sampling Execution Trigger */}
              <button
                onClick={handleSample}
                disabled={sampling}
                className="btn-pill-magenta w-full text-white py-3.5 rounded-full text-xs font-extrabold tracking-wider uppercase flex items-center justify-center gap-2.5 cursor-pointer mt-2"
              >
                {sampling ? (
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                ) : (
                  <Play className="w-4 h-4 fill-white text-white" />
                )}
                <span>RUN SAMPLING LOOP</span>
              </button>
            </div>
          </div>

          {/* Generated Passage & Image Synthesis */}
          <div className="lg:col-span-7 space-y-5">
            {genResult ? (
              <>
                {/* Text Output Deck */}
                <div className="hardware-card p-5 rounded-2xl space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/5 pb-3">
                    <span className="text-xs font-bold text-white uppercase tracking-wide">
                      Autoregressive Completion
                    </span>
                    <button
                      onClick={handleSynthesizeImage}
                      disabled={imaging}
                      className="btn-pill-magenta text-white px-4 py-2 rounded-full text-xs font-bold tracking-wide flex items-center gap-2 cursor-pointer self-start sm:self-auto"
                    >
                      {imaging ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <ImageIcon className="w-3.5 h-3.5" />
                      )}
                      <span>SYNTHESIZE IMAGE</span>
                    </button>
                  </div>

                  <div className="hardware-recess p-4 rounded-xl text-xs font-mono text-slate-200 leading-relaxed border border-white/5">
                    <span className="text-slate-400">{genResult.prompt}</span>
                    <span className="text-fuchsia-300 font-bold ml-1">{genResult.completion}</span>
                  </div>
                </div>

                {/* Image Output Card */}
                {imageData && (
                  <div className="hardware-card p-5 rounded-2xl space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-amber-400" />
                        {imageData.backend === 'procedural-latent-canvas' ? 'Procedural Latent Visual' : 'Neural Diffusion Synthesized Imagery'}
                      </span>
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold border ${
                        imageData.backend === 'procedural-latent-canvas'
                          ? 'bg-amber-950/60 text-amber-300 border-amber-700/50'
                          : 'bg-fuchsia-950/60 text-fuchsia-300 border-fuchsia-700/50'
                      }`}>
                        {imageData.backend}
                      </span>
                    </div>

                    <div className="hardware-recess p-2 rounded-2xl flex justify-center">
                      <img
                        src={imageData.image_data_url}
                        alt="synthesis visual"
                        className="rounded-xl max-w-full h-auto max-h-[360px] shadow-2xl"
                      />
                    </div>

                    {imageData.backend === 'procedural-latent-canvas' && (
                      <div className="p-3 bg-[#161122] border border-amber-500/20 rounded-xl text-[11px] text-amber-200/90 font-mono flex items-center justify-between">
                        <span>Offline procedural visual mode. To generate photorealistic diffusion images, add HF_TOKEN to Vercel Environment Variables and redeploy.</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Step Trace Table */}
                <div className="hardware-card p-5 rounded-2xl space-y-3">
                  <span className="text-xs font-bold text-white flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-fuchsia-400" />
                    Step-by-Step Autoregressive Trace
                  </span>

                  <div className="overflow-x-auto max-h-72 hardware-recess p-2 rounded-2xl">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead className="sticky top-0 bg-[#161220] text-slate-400 border-b border-white/10 font-bold">
                        <tr>
                          <th className="py-2 px-3">Step</th>
                          <th className="py-2 px-3">Token</th>
                          <th className="py-2 px-3">p(token)</th>
                          <th className="py-2 px-3">Kept</th>
                          <th className="py-2 px-3">Entropy (bits)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5 font-mono text-slate-300">
                        {genResult.steps.map((s) => (
                          <tr key={s.position} className="hover:bg-white/[0.02]">
                            <td className="py-2 px-3 text-slate-500">#{s.position}</td>
                            <td className="py-2 px-3">
                              <span className="px-2 py-0.5 rounded-md bg-[#221c2e] border border-white/10 text-fuchsia-300 font-bold">
                                {s.token_text}
                              </span>
                            </td>
                            <td className="py-2 px-3 text-[#ff8a65] font-extrabold">{(s.probability * 100).toFixed(2)}%</td>
                            <td className="py-2 px-3 text-slate-300">{s.kept_candidates.toLocaleString()}</td>
                            <td className="py-2 px-3 text-violet-400">{s.entropy_bits.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="hardware-recess p-12 rounded-3xl text-center flex flex-col items-center justify-center">
                <Play className="w-10 h-10 text-fuchsia-500/30 mb-3" />
                <span className="text-sm font-bold text-slate-300">Ready to Sample</span>
                <p className="text-xs text-slate-500 max-w-sm mt-1">
                  Adjust rotary knobs and click "RUN SAMPLING LOOP" to trace candidate probability mass.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
