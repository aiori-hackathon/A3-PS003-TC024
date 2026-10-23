export type MetricRow = { seed: number; scenario: string; mae: number; rmse: number; accuracy_rounded: number; macro_f1_rounded: number }
export type Overview = { target: string; labels: string[]; locations: number; train_rows: number; test_rows: number; feature_count: number; split: string; best_verified: Record<string, number> | null; experiments: MetricRow[]; observation_trend: {time:string;observations:number}[]; federated_available: boolean; communication_measured: boolean }
const base = import.meta.env.VITE_API_URL || 'http://localhost:8000'
async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${base}${path}`)
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `API request failed (${response.status})`) }
  return response.json()
}
export const api = {
  health: () => get<{status:string}>('/api/health'),
  overview: () => get<Overview>('/api/overview'),
  locations: async () => (await get<{locations:string[]}>('/api/locations')).locations,
  experiments: async () => (await get<{experiments:MetricRow[]}>('/api/experiments')).experiments,
  federated: () => get<{available:boolean;rounds:number|null;clients:number;reason:string}>('/api/federated/status'),
  communication: () => get<{available:boolean;reason:string}>('/api/experiments/42:centralized/communication'),
  predict: (location:string,time:string) => post<{location:string;time:string;model:string;raw_ordinal_score:number;category:string;category_index:number;basis:string}>('/api/predict',{location,time,model:'centralized'}),
}
async function post<T>(path:string, body:unknown):Promise<T>{
  const response=await fetch(`${base}${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  if(!response.ok){const data=await response.json().catch(()=>({}));throw new Error(data.detail||`API request failed (${response.status})`)}
  return response.json()
}
