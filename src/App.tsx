import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { U1Sampling } from './components/U1Sampling';
import { U2Attention } from './components/U2Attention';
import { U3BertGpt } from './components/U3BertGpt';
import { U4SamplerImage } from './components/U4SamplerImage';
import { U5ResponsibleAI } from './components/U5ResponsibleAI';
import { api } from './api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [systemHealth, setSystemHealth] = useState<{ status: string; device: string; has_hf_token?: boolean } | null>(null);

  useEffect(() => {
    api.getHealth()
      .then((res) => setSystemHealth(res))
      .catch((err) => console.warn('Health check:', err));
  }, []);

  return (
    <div className="min-h-screen bg-[#110e17] text-slate-100 flex flex-col font-['Outfit',sans-serif] relative overflow-hidden">
      {/* Ambient Underglow Lighting */}
      <div className="fixed -top-32 -left-32 w-[600px] h-[600px] rounded-full ambient-glow-peach pointer-events-none z-0 opacity-60 blur-3xl" />
      <div className="fixed top-1/3 -right-32 w-[650px] h-[650px] rounded-full ambient-glow-magenta pointer-events-none z-0 opacity-55 blur-3xl" />
      <div className="fixed -bottom-40 left-1/3 w-[500px] h-[500px] rounded-full ambient-glow-peach pointer-events-none z-0 opacity-40 blur-3xl" />

      {/* Main Console Interface */}
      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} systemHealth={systemHealth} />

        <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 lg:px-8 py-4 sm:py-6">
          {activeTab === 0 && <U1Sampling />}
          {activeTab === 1 && <U2Attention />}
          {activeTab === 2 && <U3BertGpt />}
          {activeTab === 3 && <U4SamplerImage />}
          {activeTab === 4 && <U5ResponsibleAI />}
        </main>

        <footer className="mt-8 sm:mt-12 py-6 sm:py-8 border-t border-white/5 bg-[#0d0a13]/90 backdrop-blur-md text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 flex-wrap justify-center">
              <span className="w-2 h-2 rounded-full bg-fuchsia-500 shadow-[0_0_8px_#d946ef]" />
              <span className="font-mono text-slate-400 font-bold">NEXT-TOKEN PROBABILITY LAB</span>
              <span className="text-slate-600">&bull;</span>
              <span className="text-slate-400">TRACK A &bull; PROBLEM P5</span>
            </div>

            <div className="flex items-center gap-2 sm:gap-3 font-mono text-[10px] sm:text-[11px] text-slate-400 flex-wrap justify-center">
              <span className="px-2.5 py-1 rounded bg-[#171322] border border-white/5">PyTorch 2.x</span>
              <span className="px-2.5 py-1 rounded bg-[#171322] border border-white/5">Hugging Face Transformers</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
