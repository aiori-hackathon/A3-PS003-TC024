export type MetricRow = {
  seed: number
  scenario: string
  mae: number
  rmse: number
  accuracy_rounded: number
  macro_f1_rounded: number
}

export type Overview = {
  target: string
  labels: string[]
  locations: number
  train_rows: number
  test_rows: number
  feature_count: number
  split: string
  observed_time_range: { start: string; end: string }
  best_verified: Record<string, number | string | null> | null
  experiments: MetricRow[]
  observation_trend: { time: string; observations: number }[]
  federated_available: boolean
  federated_rounds: number
  federated_partitions: string[]
  fedavg_summary: Record<string, number | string | null>[]
  saved_fedavg_runs: number
  communication_measured: boolean
}

export type ApiError = Error & { endpoint: string; status?: number }
const base = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const endpoint = `${base}${path}`
  let response: Response
  try {
    response = await fetch(endpoint, init)
  } catch (cause) {
    const error = new Error(`Could not reach ${endpoint}. Check the API server and browser CORS settings.`) as ApiError
    error.endpoint = endpoint
    throw error
  }

  const contentType = response.headers.get('content-type') || ''
  let payload: unknown
  try {
    payload = contentType.includes('application/json') ? await response.json() : await response.text()
  } catch {
    const error = new Error(`${init?.method || 'GET'} ${path} returned invalid JSON (HTTP ${response.status}).`) as ApiError
    error.endpoint = endpoint
    error.status = response.status
    throw error
  }

  if (!response.ok) {
    const detail = typeof payload === 'object' && payload !== null && 'detail' in payload
      ? String((payload as { detail: unknown }).detail)
      : typeof payload === 'string' && payload ? payload : response.statusText
    const error = new Error(`${init?.method || 'GET'} ${path} failed (HTTP ${response.status}): ${detail}`) as ApiError
    error.endpoint = endpoint
    error.status = response.status
    throw error
  }
  if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
    const error = new Error(`${init?.method || 'GET'} ${path} returned an unexpected response shape.`) as ApiError
    error.endpoint = endpoint
    error.status = response.status
    throw error
  }
  return payload as T
}

export const api = {
  health: () => request<{ status: string; dataset_available: boolean; results_available: boolean }>('/api/health'),
  overview: async () => {
    const result = await request<Overview>('/api/overview')
    if (!Array.isArray(result.observation_trend) || typeof result.locations !== 'number' || !('best_verified' in result)) {
      throw new Error('GET /api/overview returned an incomplete dashboard payload.')
    }
    return result
  },
  locations: async () => {
    const result = await request<{ locations: string[] }>('/api/locations')
    if (!Array.isArray(result.locations)) throw new Error('GET /api/locations returned an invalid locations list.')
    return result.locations
  },
  experiments: async () => {
    const result = await request<{ experiments: MetricRow[]; summaries: Record<string, unknown>[] }>('/api/experiments')
    if (!Array.isArray(result.experiments)) throw new Error('GET /api/experiments returned an invalid experiments list.')
    return result.experiments
  },
  federated: () => request<{ available: boolean; rounds: number | null; clients: number; reason: string }>('/api/federated/status'),
  communication: () => request<{
    available: boolean; measured: boolean; reason: string; rounds?: number; clients?: number; model_bytes?: number
    client_to_server_bytes_per_round?: number; server_to_client_bytes_per_round?: number
    total_bytes_per_round?: number; total_bytes?: number
  }>('/api/experiments/42:fedavg_iid/communication'),
  predict: (location: string, time: string) => request<{
    location: string; time: string; model: string; raw_ordinal_score: number; category: string
    category_index: number; basis: string
  }>('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ location, time, model: 'centralized' }) }),
}
