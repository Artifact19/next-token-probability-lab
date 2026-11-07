// API Client for Next-Token Probability Lab

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export interface DistributionResponse {
  tokens: string[];
  token_ids: number[];
  raw_probs: number[];
  adjusted_probs: number[];
  raw_entropy_bits: number;
  adjusted_entropy_bits: number;
  kept_candidates: number;
  nucleus_curve: number[];
  top_p: number;
}

export interface CompareTempsResponse {
  tokens: string[];
  probs_low: number[];
  probs_high: number[];
  text_low: string;
  text_high: string;
}

export interface AttentionResponse {
  tokens: string[];
  matrix: number[][];
  received_attention: number[];
  num_layers: number;
  num_heads: number;
  selected_layer: number;
  selected_head: number | null;
}

export interface PerplexityResponse {
  token_count: number;
  gpt_ppl: number;
  lstm_ppl: number | null;
  has_lstm: boolean;
  block_size: number;
}

export interface BertCompareResponse {
  gpt2_top: { word: string; probability: number }[];
  bert_top: { word: string; probability: number }[];
  aligned_rows: { word: string; gpt2_prob: number; bert_prob: number }[];
  jaccard_overlap: number;
}

export interface SamplingStep {
  position: number;
  token_id: number;
  token_text: string;
  probability: number;
  kept_candidates: number;
  entropy_bits: number;
}

export interface GenerateResponse {
  prompt: string;
  completion: string;
  text: string;
  steps: SamplingStep[];
}

export interface ImageSynthResponse {
  image_data_url: string;
  backend: string;
}

export interface AuditReport {
  temperature: number;
  trials: number;
  error_rate: number;
  error_ci: [number, number];
  hallucination_rate: number;
  hallucination_ci: [number, number];
  mean_entropy_bits: number;
  failures: { question: string; expected: string; generated: string }[];
}

export interface ToxicSuppressionResponse {
  unfiltered: string;
  filtered: string;
  peak_suppressed_mass: number;
  suppressed_mass_history: number[];
  blocklist: string[];
}

export interface CalibrationResponse {
  ece: number;
  bin_conf: number[];
  bin_acc: number[];
  bin_count: number[];
}

async function post<T>(endpoint: string, body: any): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || `API Error: ${res.statusText}`);
  }
  return res.json();
}

async function get<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API Error: ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  getHealth: () => get<{ status: string; device: string; has_hf_token: boolean; lstm_checkpoint_exists: boolean }>('/health'),
  
  getDistribution: (params: { prompt: string; temperature: number; top_k: number; top_p: number; top_n: number }) =>
    post<DistributionResponse>('/u1/distribution', params),
    
  compareTemperatures: (params: { prompt: string; top_k: number; top_p: number; top_n: number; seed: number }) =>
    post<CompareTempsResponse>('/u1/compare-temperatures', params),
    
  getAttention: (params: { prompt: string; layer: number; head?: number | null }) =>
    post<AttentionResponse>('/u2/attention', params),
    
  getPerplexity: (params: { source: string; custom_text?: string; block_size: number }) =>
    post<PerplexityResponse>('/u2/perplexity', params),
    
  compareBertGpt: (params: { masked_text: string; use_right_context: boolean; top_n: number }) =>
    post<BertCompareResponse>('/u3/compare', params),
    
  generateWithTrace: (params: { prompt: string; max_new_tokens: number; temperature: number; top_k: number; top_p: number; seed: number }) =>
    post<GenerateResponse>('/u4/generate', params),
    
  synthesizeImage: (params: { text: string; seed: number }) =>
    post<ImageSynthResponse>('/u4/synthesize-image', params),
    
  runAudit: (params: { temperatures: number[]; samples_per_item: number; seed?: number }) =>
    post<{ reports: AuditReport[] }>('/u5/audit', params),
    
  testToxicFilter: (params: { prompt: string; temperature: number; seed: number; max_tokens: number }) =>
    post<ToxicSuppressionResponse>('/u5/toxic-suppression', params),
    
  getCalibration: () =>
    post<CalibrationResponse>('/u5/calibration', {}),
    
  getModelCard: () =>
    get<{ content: string }>('/model-card'),
};

