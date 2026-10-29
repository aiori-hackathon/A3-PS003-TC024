import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Activity, ArrowRight, BarChart3, BookOpen, CheckCircle2, ChevronDown, CircleHelp,
  Clock3, Cpu, Database, Gauge, GitFork, Info, Menu, Network, RefreshCw, Route,
  Sparkles, Users, X,
} from 'lucide-react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, type MetricRow, type Overview } from './api'

type Page = 'Overview' | 'Predictions' | 'Model Lab' | 'Federated Learning' | 'Communication' | 'Experiments'
type RequestState = 'loading' | 'success' | { error: string }
type ScenarioSummary = { scenario: string; rows: MetricRow[]; mae?: number; accuracy?: number; rmse?: number; macroF1?: number }

const nav: { name: Page; icon: typeof Activity; hint: string }[] = [
  { name: 'Overview', icon: Gauge, hint: 'The study at a glance' },
  { name: 'Predictions', icon: Route, hint: 'Try a historical estimate' },
  { name: 'Model Lab', icon: BarChart3, hint: 'Compare model results' },
  { name: 'Federated Learning', icon: Network, hint: 'How learners work together' },
  { name: 'Communication', icon: Activity, hint: 'Estimated model exchange' },
  { name: 'Experiments', icon: BookOpen, hint: 'Saved evaluation results' },
]

const scenarios = [
  'centralized', 'location_mean_baseline', 'train_mean_baseline',
  'fedavg_iid', 'fedavg_location', 'local_iid_client_average', 'local_location_client_average',
]

const scenarioNames: Record<string, string> = {
  centralized: 'Centralized model',
  location_mean_baseline: 'Location mean baseline',
  train_mean_baseline: 'Training mean baseline',
  fedavg_iid: 'FedAvg · IID',
  fedavg_location: 'FedAvg · location-based',
  local_iid_client_average: 'Local-only · IID',
  local_location_client_average: 'Local-only · location-based',
}

const fmt = (value: unknown, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Not available'
const pct = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not available'
const uniqueRuns = (rows: MetricRow[]) => [...new Map(rows.map(row => [`${row.seed}:${row.scenario}`, row])).values()]

function summarizeRuns(rows: MetricRow[]): ScenarioSummary[] {
  const unique = uniqueRuns(rows)
  return scenarios.map(scenario => {
    const group = unique.filter(row => row.scenario === scenario)
    const mean = (key: keyof MetricRow) => {
      const values = group.map(row => row[key]).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : undefined
    }
    return { scenario, rows: group, mae: mean('mae'), rmse: mean('rmse'), accuracy: mean('accuracy_rounded'), macroF1: mean('macro_f1_rounded') }
  })
}

function scenarioName(scenario: string) { return scenarioNames[scenario] ?? scenario.replaceAll('_', ' ') }

export function App() {
  const [page, setPage] = useState<Page>('Overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [data, setData] = useState<Overview | null>(null)
  const [runs, setRuns] = useState<MetricRow[]>([])
  const [health, setHealth] = useState<RequestState>('loading')
  const [overviewState, setOverviewState] = useState<RequestState>('loading')
  const [experimentState, setExperimentState] = useState<RequestState>('loading')

  async function loadDashboard() {
    setHealth('loading')
    setOverviewState('loading')
    setExperimentState('loading')
    void api.health()
      .then(result => setHealth(result.status === 'ok' ? 'success' : { error: `The health endpoint returned “${result.status}”.` }))
      .catch(error => setHealth({ error: error instanceof Error ? error.message : String(error) }))
    void api.overview()
      .then(result => { setData(result); setOverviewState('success') })
      .catch(error => setOverviewState({ error: error instanceof Error ? error.message : String(error) }))
    void api.experiments()
      .then(result => { setRuns(uniqueRuns(result)); setExperimentState('success') })
      .catch(error => setExperimentState({ error: error instanceof Error ? error.message : String(error) }))
  }

  useEffect(() => { void loadDashboard() }, [])
  const summaries = useMemo(() => summarizeRuns(runs), [runs])
  const allLoaded = health === 'success' && overviewState === 'success' && experimentState === 'success'
  const failures = [overviewState, experimentState].filter((state): state is { error: string } => typeof state === 'object')
  const apiReachable = health === 'success' || overviewState === 'success' || experimentState === 'success' ||
    (typeof health === 'object' && (health.error.includes('HTTP') || health.error.includes('returned')))

  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`} aria-label="Main navigation">
        <div className="brand-lockup">
          <SynRouteMark />
          <div><strong>SynRoute</strong><span>FEDERATED TRAFFIC AI</span></div>
          <button className="icon-button mobile-close" onClick={() => setMenuOpen(false)} aria-label="Close navigation"><X size={19} /></button>
        </div>
        <div className="nav-caption">EXPLORE</div>
        <nav className="primary-nav">
          {nav.map(item => {
            const Icon = item.icon
            return <button key={item.name} className={`nav-item ${page === item.name ? 'nav-selected' : ''}`} onClick={() => { setPage(item.name); setMenuOpen(false) }} aria-current={page === item.name ? 'page' : undefined}>
              <Icon size={19} aria-hidden="true" /><span><b>{item.name}</b><small>{item.hint}</small></span>{page === item.name && <i aria-hidden="true" />}
            </button>
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-study"><div className="study-icon"><Database size={17} /></div><div><strong>Pune traffic study</strong><span>Historical observations · 4 classes</span></div></div>
          <p className="sidebar-foot">A learning project using saved data and simulated clients.</p>
        </div>
      </aside>
      {menuOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}

      <main className="main-content">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMenuOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
          <div className="breadcrumb"><span>SynRoute</span><b>/</b>{page}</div>
          <div className="topbar-right">
            <div className={`service-status ${apiReachable ? 'is-online' : ''}`} title={health === 'success' && allLoaded ? 'API and dashboard data loaded successfully' : 'Health status is separate from dashboard data requests'}>
              <i aria-hidden="true" />{health === 'loading' ? 'Checking connection' : health !== 'success' && !apiReachable ? 'API unavailable' : allLoaded ? 'Data ready' : failures.length ? 'Some data unavailable' : 'API reachable · loading data'}
            </div>
            <span className="historical-pill"><span />Historical data</span>
          </div>
        </header>

        <section className="page-wrap">
          {typeof health === 'object' && <RequestBanner title="The API health check failed" message={health.error} onRetry={loadDashboard} />}
          {failures.length > 0 && <div className="request-banner" role="alert"><div><strong>Some dashboard information could not be loaded</strong>{failures.map((failure, index) => <p key={index}>{failure.error}</p>)}</div><button className="button button-secondary" onClick={loadDashboard}><RefreshCw size={16} />Retry data requests</button></div>}
          {page !== 'Overview' && <PageHeader page={page} />}
          {page === 'Overview' && <OverviewPage data={data} summaries={summaries} overviewState={overviewState} experimentState={experimentState} onRetry={loadDashboard} onNavigate={setPage} />}
          {page === 'Predictions' && <PredictionPage labels={data?.labels ?? []} />}
          {page === 'Model Lab' && <ModelLabPage data={data} summaries={summaries} runs={runs} />}
          {page === 'Federated Learning' && <FederatedPage summaries={summaries} />}
          {page === 'Communication' && <CommunicationPage />}
          {page === 'Experiments' && <ExperimentsPage data={data} summaries={summaries} runs={runs} state={experimentState} onRetry={loadDashboard} />}
          <footer className="app-footer"><span>SynRoute · Federated AI for Traffic Intelligence</span><span>Historical Pune data · no live traffic feed</span></footer>
        </section>
      </main>
      <nav className="mobile-tabbar" aria-label="Primary navigation">
        {nav.map(item => {
          const Icon = item.icon
          return <button key={item.name} className={page === item.name ? 'mobile-tab-active' : ''} onClick={() => setPage(item.name)} aria-current={page === item.name ? 'page' : undefined}>
            <Icon size={19} aria-hidden="true" /><span>{item.name === 'Federated Learning' ? 'Federated' : item.name}</span>
          </button>
        })}
      </nav>
    </div>
  )
}

function SynRouteMark({ large = false }: { large?: boolean }) {
  return <span className={`synroute-mark ${large ? 'mark-large' : ''}`} aria-hidden="true">
    <svg viewBox="0 0 48 48" fill="none"><path d="M8 35 19 24l8 8 13-17" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /><path d="M30 15h10v10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /><circle cx="8" cy="35" r="3" fill="currentColor" /><circle cx="19" cy="24" r="3" fill="currentColor" /><circle cx="27" cy="32" r="3" fill="currentColor" /></svg>
  </span>
}

function PageHeader({ page }: { page: Page }) {
  if (page === 'Overview') return null
  const copy: Record<Page, { eyebrow: string; title: string; description: string }> = {
    Overview: { eyebrow: 'PUNE TRAFFIC STUDY', title: 'Understand traffic. Predict congestion.', description: 'See what historical observations can teach us about congestion—and how different models performed.' },
    Predictions: { eyebrow: 'TRY A MODEL ESTIMATE', title: 'Explore a congestion estimate', description: 'Choose a known origin and time. SynRoute estimates a class from historical training data; it does not report current road conditions.' },
    'Model Lab': { eyebrow: 'MODEL COMPARISON', title: 'How did the models compare?', description: 'Compare prediction errors and class accuracy across the saved models and simple reference baselines.' },
    'Federated Learning': { eyebrow: 'LEARNING TOGETHER', title: 'One model, many learners', description: 'See how simulated data clients train locally and combine model updates over saved FedAvg rounds.' },
    Communication: { eyebrow: 'MODEL EXCHANGE', title: 'What gets sent between learners?', description: 'Explore payload-size estimates for a model update and the repeated client–server exchange.' },
    Experiments: { eyebrow: 'SAVED EVALUATIONS', title: 'Explore the experiment results', description: 'Review the saved seed-by-seed results, how each method was tested, and what they can—and cannot—tell us.' },
  }
  const content = copy[page]
  return <div className="page-heading"><span className="eyebrow">{content.eyebrow}</span><h1>{content.title}</h1><p>{content.description}</p></div>
}

function RequestBanner({ title, message, onRetry }: { title: string; message: string; onRetry: () => void }) {
  return <div className="request-banner" role="alert"><div><strong>{title}</strong><p>{message}</p></div><button className="button button-secondary" onClick={onRetry}><RefreshCw size={16} />Retry</button></div>
}

function RequestFailure({ state, onRetry }: { state: { error: string }; onRetry: () => void }) {
  return <div className="request-failure" role="alert"><CircleHelp size={20} /><div><strong>This section could not be loaded</strong><p>{state.error}</p></div><button className="button button-secondary" onClick={onRetry}><RefreshCw size={16} />Try again</button></div>
}

function SectionHeading({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</div>
}

function Explain({ title, children }: { title: string; children: ReactNode }) {
  return <details className="explain"><summary><Info size={15} />{title}</summary><div>{children}</div></details>
}

function MetricCard({ icon: Icon, label, value, description, tone, explanation }: {
  icon: typeof Activity; label: string; value: string; description: string; tone?: string; explanation?: ReactNode
}) {
  return <article className={`metric-card ${tone ?? ''}`}>
    <div className="metric-card-label"><Icon size={18} aria-hidden="true" /><span>{label}</span></div>
    <strong className="metric-card-value">{value}</strong>
    <p>{description}</p>
    {explanation && <Explain title="How to read this">{explanation}</Explain>}
  </article>
}

function OverviewPage({ data, summaries, overviewState, experimentState, onRetry, onNavigate }: {
  data: Overview | null; summaries: ScenarioSummary[]; overviewState: RequestState; experimentState: RequestState
  onRetry: () => void; onNavigate: (page: Page) => void
}) {
  const central = summaries.find(row => row.scenario === 'centralized')
  const accuracy = numberValue(data?.best_verified?.accuracy_rounded_mean) ?? central?.accuracy
  const mae = numberValue(data?.best_verified?.mae_mean) ?? central?.mae
  const bestMae = summaries.filter(row => row.mae !== undefined).reduce<ScenarioSummary | null>((best, row) => !best || (row.mae ?? Infinity) < (best.mae ?? Infinity) ? row : best, null)
  const fedavgRows = [summaries.find(row => row.scenario === 'fedavg_iid'), summaries.find(row => row.scenario === 'fedavg_location')].filter((row): row is ScenarioSummary => Boolean(row?.rows.length))
  const fedavgBelowCentral = central?.mae !== undefined && central.accuracy !== undefined && fedavgRows.length > 0 && fedavgRows.every(row => row.mae !== undefined && row.accuracy !== undefined && row.mae > central.mae! && row.accuracy < central.accuracy!)
  const trend = data?.observation_trend ?? []
  const peak = trend.reduce<(typeof trend)[number] | null>((best, current) => !best || current.observations > best.observations ? current : best, null)
  const timeRange = data?.observed_time_range
  const learned = bestMae?.mae !== undefined && mae !== undefined && accuracy !== undefined
    ? `In these saved evaluations, ${scenarioName(bestMae.scenario)} had the lowest average error (${fmt(bestMae.mae)}). The centralized model matched ${pct(accuracy)} of rounded congestion classes${fedavgBelowCentral ? ', while both saved FedAvg splits scored lower on accuracy and higher on error' : ''}.`
    : 'Saved comparisons will appear here when the experiment metrics are available.'

  return <div className="page-content">
    <section className="hero-panel">
      <div className="hero-copy"><div className="hero-brand"><SynRouteMark large /><span>SynRoute</span><i>Many learners. One smarter network.</i></div>
        <h1>See how traffic moves.<br />Understand how AI learns.</h1>
        <p>SynRoute studies historical traffic patterns and compares AI models to explore congestion prediction and collaborative learning.</p>
        <div className="hero-tags"><span><Clock3 size={15} />Historical records{timeRange ? ` · ${timeRange.start.slice(0, 5)}–${timeRange.end.slice(0, 5)}` : ''}</span><span><Users size={15} />Simulated learning clients</span></div>
      </div>
      <div className="hero-network" aria-hidden="true"><svg viewBox="0 0 420 230" role="presentation"><path d="M40 175 125 130l70 34 86-96 93 31M125 130l22-87 112 25 22 96 93 31M195 164l-48 38m134-134 93-51" /><circle cx="40" cy="175" r="8"/><circle cx="125" cy="130" r="9"/><circle cx="147" cy="43" r="7"/><circle cx="195" cy="164" r="8"/><circle cx="283" cy="68" r="11"/><circle cx="305" cy="164" r="7"/><circle cx="374" cy="99" r="8"/><circle cx="397" cy="15" r="6"/></svg><span className="network-caption">One shared learning goal</span></div>
    </section>

    <div className="section-intro"><div><span className="section-kicker">THE STUDY</span><h2>What did we study?</h2><p>Separate records teach the model and check how well it learned.</p></div><span className="data-badge"><CheckCircle2 size={16} />Historical dataset</span></div>
    {typeof overviewState === 'object' ? <RequestFailure state={overviewState} onRetry={onRetry} /> : null}
    {overviewState === 'loading' && <LoadingCard label="Loading the traffic data summary…" />}
    {overviewState === 'success' && data && <>
      <section className="data-grid">
        <MetricCard icon={Database} label="Examples used to teach" value={data.train_rows.toLocaleString()} description="Training records: observations the models learned from." explanation={<>These cleaned traffic records formed the training split. The model uses time of day and origin information to learn congestion patterns.</>} />
        <MetricCard icon={CheckCircle2} label="Examples used to check" value={data.test_rows.toLocaleString()} description="Test records: separate observations used for evaluation." explanation={<>The test split is held aside during training and used to calculate the saved evaluation metrics.</>} />
        <MetricCard icon={Route} label="Traffic origins" value={data.locations.toLocaleString()} description="Unique origin IDs present in the training data." explanation={<>This counts distinct <code>stop_id_from</code> values in the training split. It is not a count of roads, intersections, or all locations in Pune.</>} />
        <MetricCard icon={Clock3} label="Observed training times" value={timeRange ? `${timeRange.start.slice(0, 5)}–${timeRange.end.slice(0, 5)}` : 'Not available'} description={`${trend.length} quarter-hour buckets with recorded examples.`} explanation={<>The chart groups training records into 15-minute buckets. Empty buckets are omitted; this is not live traffic volume.</>} />
      </section>
      <details className="technical-details"><summary>Technical details about the data</summary><div className="detail-columns"><p><b>Inputs</b><br />Time represented as a repeating daily cycle, plus one-hot origin IDs.</p><p><b>Target</b><br />Congestion class on an ordinal 0–3 scale.</p><p><b>Evaluation split</b><br />{data.split}. The dataset contains no calendar date.</p><p><b>Encoded input count</b><br />{data.feature_count.toLocaleString()} features.</p></div></details>
    </>}

    <div className="section-intro section-spaced"><div><span className="section-kicker">SAVED EVALUATION</span><h2>How well does the AI predict?</h2><p>These are averages across the saved centralized-model seeds.</p></div><button className="text-button" onClick={() => onNavigate('Model Lab')}>Compare all models <ArrowRight size={16} /></button></div>
    {typeof experimentState === 'object' ? <RequestFailure state={experimentState} onRetry={onRetry} /> : experimentState === 'loading' ? <LoadingCard label="Loading saved model results…" /> : <section className="performance-grid">
      <MetricCard icon={Gauge} tone="metric-blue" label="Rounded class accuracy" value={pct(accuracy)} description={typeof accuracy === 'number' ? `About ${Math.round(accuracy * 100)} of every 100 test predictions matched the target class.` : 'Saved accuracy is not available.'} explanation={<>The model’s numeric score is rounded to the nearest class and clipped to 0–3. Accuracy is the share of test records where that class matches the recorded target.</>} />
      <MetricCard icon={Activity} tone="metric-teal" label="Average class-level error (MAE)" value={fmt(mae)} description="Average distance between predicted and actual congestion levels." explanation={<>MAE is the mean absolute difference between the raw model score and the actual ordinal label. A smaller value means predictions are closer on the 0–3 scale. It does not represent minutes or time spent in traffic.</>} />
    </section>}

    <section className="chart-section surface-card">
      <SectionHeading title="What does the data show?" description="Historical training records grouped by time of day." action={<span className="chart-key"><i />Recorded examples</span>} />
      {typeof overviewState === 'object' ? <RequestFailure state={overviewState} onRetry={onRetry} /> : overviewState === 'loading' ? <LoadingCard label="Loading observation counts…" /> : trend.length > 0 ? <>
        <div className="observation-chart" role="img" aria-label={`Historical training records grouped by 15-minute time bucket. ${peak ? `The largest bucket was ${peak.time} with ${peak.observations} records.` : ''}`}>
          <ResponsiveContainer width="100%" height="100%"><AreaChart data={trend} margin={{ top: 12, right: 12, left: 8, bottom: 28 }}>
            <defs><linearGradient id="trafficFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#da4f38" stopOpacity={0.32} /><stop offset="95%" stopColor="#da4f38" stopOpacity={0.02} /></linearGradient></defs>
            <CartesianGrid stroke="#d8d1c4" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: '#596476', fontSize: 13 }} tickLine={false} axisLine={false} minTickGap={24} label={{ value: 'Time of day', position: 'insideBottom', offset: -17, fill: '#596476', fontSize: 14 }} />
            <YAxis tick={{ fill: '#596476', fontSize: 13 }} tickLine={false} axisLine={false} width={56} label={{ value: 'Records', angle: -90, position: 'insideLeft', fill: '#596476', fontSize: 14 }} />
            <Tooltip contentStyle={{ background: '#fffdf8', border: '1px solid #d8d1c4', borderRadius: 0, color: '#172033', fontSize: 14 }} formatter={(value) => [Number(value).toLocaleString(), 'Training records']} labelFormatter={label => `${label} · 15-minute bucket`} />
            <Area type="monotone" dataKey="observations" stroke="#da4f38" strokeWidth={3} fill="url(#trafficFill)" activeDot={{ r: 5, fill: '#155eef', stroke: '#fffdf8', strokeWidth: 2 }} />
          </AreaChart></ResponsiveContainer>
        </div>
        <div className="chart-summary"><Sparkles size={18} /><p>{peak ? `The largest recorded training bucket was ${peak.time}, with ${peak.observations.toLocaleString()} examples.` : 'Training observations are grouped by time of day.'} Counts describe the dataset, not how busy roads are now.</p></div>
        <details className="technical-details compact-details"><summary>About the chart</summary><p>Each point counts rows in the training split, grouped into 15-minute intervals. Missing intervals are not filled with zero. The source data has times but no dates.</p></details>
      </> : <EmptyState title="No observation counts available" text="The API returned no time buckets for the training split." />}
    </section>

    <section className="learning-summary surface-card">
      <div className="summary-mark"><Sparkles size={21} /></div><div><span className="section-kicker">WHAT SYNROUTE LEARNED</span><h2>A useful comparison, not a promise.</h2><p>{learned}</p><p className="caveat-line">These results come from a historical time-of-day holdout. They do not establish live-road accuracy or future-date performance.</p></div>
      <button className="button button-secondary summary-link" onClick={() => onNavigate('Model Lab')}>See model comparison <ArrowRight size={16} /></button>
    </section>

    <section className="limitations-card">
      <CircleHelp size={19} /><div><strong>What this study does not tell us</strong><p>There are no calendar dates or live traffic feeds. This experiment does not measure real-world deployments, network transfers, or the privacy of model updates.</p></div>
      <Explain title="More about the limits">The holdout is based on time of day within each origin, not a test on later calendar dates. Predictions are estimates for supplied inputs, not verified road conditions.</Explain>
    </section>
  </div>
}

function LoadingCard({ label }: { label: string }) { return <div className="loading-card" role="status"><span className="spinner" /><span>{label}</span></div> }
function EmptyState({ title, text }: { title: string; text: string }) { return <div className="empty-state"><Database size={22} /><strong>{title}</strong><p>{text}</p></div> }

function PredictionPage({ labels }: { labels: string[] }) {
  const [locations, setLocations] = useState<string[]>([])
  const [locationsState, setLocationsState] = useState<RequestState>('loading')
  const [location, setLocation] = useState('')
  const [time, setTime] = useState('15:00')
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.predict>> | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function loadLocations() {
    setLocationsState('loading')
    try { const values = await api.locations(); setLocations(values); if (values.length && !location) setLocation(values[0]); setLocationsState('success') }
    catch (reason) { setLocationsState({ error: reason instanceof Error ? reason.message : String(reason) }) }
  }
  useEffect(() => { void loadLocations() }, [])

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setResult(null); setSubmitting(true)
    try { setResult(await api.predict(location, time)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setSubmitting(false) }
  }

  return <div className="page-content">
    <section className="prediction-intro surface-card"><div className="step-bubble">1</div><div><h2>Choose a place and time</h2><p>The model uses a known traffic origin and time of day to estimate one of the four congestion classes.</p></div><div className="historical-pill"><span />Historical input only</div></section>
    <div className="prediction-layout">
      <section className="surface-card prediction-form-card">
        <SectionHeading title="Try a model estimate" description="This is a model output, not a live road reading." />
        {typeof locationsState === 'object' ? <RequestFailure state={locationsState} onRetry={loadLocations} /> : locationsState === 'loading' ? <LoadingCard label="Loading traffic origins…" /> : locations.length === 0 ? <EmptyState title="No origins available" text="The API returned no traffic origins for this dataset." /> : <form className="prediction-form" onSubmit={submit}>
          <label htmlFor="origin-select">Traffic origin <span>Choose one of the origins represented in training data.</span></label>
          <select id="origin-select" value={location} onChange={event => setLocation(event.target.value)} required>
            {locations.map(value => <option key={value} value={value}>Origin {value}</option>)}
          </select>
          <label htmlFor="time-input">Time of day <span>Use the local time format found in the historical data.</span></label>
          <input id="time-input" type="time" value={time} onChange={event => setTime(event.target.value)} required />
          <div className="model-choice"><Cpu size={19} /><div><strong>Centralized congestion model</strong><span>Seed 42 · trained on the saved training split</span></div></div>
          <button className="button button-primary" type="submit" disabled={submitting || !location}>{submitting ? <><span className="spinner spinner-dark" />Estimating…</> : <>Estimate congestion <ArrowRight size={17} /></>}</button>
        </form>}
        <Explain title="What information does the model use?">The model receives the selected time represented as a repeating daily cycle and the chosen origin ID. It does not use live speed or a traffic feed.</Explain>
      </section>
      <section className="surface-card prediction-output" aria-live="polite">
        <SectionHeading title="Your estimate" description="The result for the inputs you selected." />
        {error ? <div className="prediction-error" role="alert"><CircleHelp size={20} /><strong>We couldn’t make that estimate.</strong><p>{error}</p><button className="button button-secondary" onClick={() => { setError(''); void loadLocations() }}>Check origins again</button></div>
          : result ? <div className="prediction-result"><span className="result-label">PREDICTED CONGESTION CLASS</span><strong className={`class-badge class-${result.category_index}`}>{result.category}</strong><div className="ordinal-score"><span>Model score on the 0–3 scale</span><b>{result.raw_ordinal_score.toFixed(2)} <small>/ 3</small></b></div><div className="class-scale" aria-label="Congestion scale from very smooth to heavy congestion">{labels.map((label, index) => <span key={label} className={index === result.category_index ? 'scale-active' : ''}>{index}<small>{label}</small></span>)}</div><div className="prediction-context"><p><b>Origin</b> {result.location}</p><p><b>Time</b> {result.time}</p><p><b>Actual class</b> Not available for a custom input</p></div><div className="plain-note"><Info size={17} /><p>This estimate is not a confirmed current road condition. The model does not calculate a confidence score.</p></div></div>
            : <div className="prediction-placeholder"><Sparkles size={25} /><strong>Ready to explore</strong><p>Choose an origin and time, then request an estimate. Nothing is predicted until you submit the form.</p></div>}
      </section>
    </div>
    <section className="steps-strip" aria-label="How this estimate is made"><div><span>1</span><p>Choose an origin and time</p></div><ArrowRight size={16} /><div><span>2</span><p>The model estimates a class</p></div><ArrowRight size={16} /><div><span>3</span><p>Read the result as an estimate</p></div></section>
  </div>
}

function ModelLabPage({ data, summaries, runs }: { data: Overview | null; summaries: ScenarioSummary[]; runs: MetricRow[] }) {
  const central = summaries.find(row => row.scenario === 'centralized')
  const location = summaries.find(row => row.scenario === 'location_mean_baseline')
  const highestAccuracy = summaries.filter(row => row.accuracy !== undefined).reduce<ScenarioSummary | null>((best, row) => !best || (row.accuracy ?? -1) > (best.accuracy ?? -1) ? row : best, null)
  const bestError = summaries.filter(row => row.mae !== undefined).reduce<ScenarioSummary | null>((best, row) => !best || (row.mae ?? Infinity) < (best.mae ?? Infinity) ? row : best, null)
  const fedavgScenarios = summaries.filter(row => row.scenario.startsWith('fedavg_') && row.rows.length > 0)
  const fedavgCompared = central?.mae !== undefined && central.accuracy !== undefined && fedavgScenarios.length > 0
  const fedavgComparison = fedavgCompared && fedavgScenarios.every(row => row.mae !== undefined && row.accuracy !== undefined && row.mae > central.mae! && row.accuracy < central.accuracy!)
    ? 'Both saved FedAvg partitions had higher MAE and lower rounded accuracy than centralized training.'
    : 'FedAvg results are shown separately because performance can differ by metric and partition.'
  const seeds = [...new Set(uniqueRuns(runs).map(row => row.seed))].sort((a, b) => a - b)
  const fedavgConfig = data?.fedavg_summary.find(row => row.scenario === 'fedavg_iid')
  const fedavgClients = numberValue(fedavgConfig?.n_clients_mean)
  const fedavgRounds = numberValue(fedavgConfig?.rounds_mean)
  const chartRows = summaries.filter(row => row.mae !== undefined).map(row => ({ name: scenarioName(row.scenario), mae: row.mae, scenario: row.scenario }))

  return <div className="page-content">
    <section className="insight-banner"><Sparkles size={21} /><div><strong>There isn’t one “best” model for every measure.</strong><p>{highestAccuracy ? `${scenarioName(highestAccuracy.scenario)} had the highest rounded accuracy (${pct(highestAccuracy.accuracy)}).` : 'Accuracy results are not available.'} {bestError ? `${scenarioName(bestError.scenario)} had the lowest MAE (${fmt(bestError.mae)}).` : ''}</p></div></section>
    <div className="model-grid">
      <section className="surface-card model-chart-card"><SectionHeading title="Average prediction error" description="Mean absolute error across saved seeds. Shorter bars mean a smaller average difference on the 0–3 class scale." />
        {chartRows.length > 0 ? <div className="model-chart" role="img" aria-label="Bar chart comparing mean absolute error across saved model scenarios"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartRows} layout="vertical" margin={{ top: 8, right: 20, left: 14, bottom: 8 }}><CartesianGrid stroke="#d8d1c4" horizontal={false} /><XAxis type="number" domain={[0, 'dataMax + 0.15']} tick={{ fill: '#596476', fontSize: 13 }} axisLine={false} tickLine={false} /><YAxis type="category" dataKey="name" width={190} tick={{ fill: '#172033', fontSize: 13 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: '#fffdf8', border: '1px solid #d8d1c4', borderRadius: 0, color: '#172033', fontSize: 14 }} formatter={(value) => [fmt(Number(value)), 'Average class-level error']} /><Bar dataKey="mae" radius={[0, 0, 0, 0]}>{chartRows.map(row => <Cell key={row.scenario} fill={row.scenario === 'centralized' ? '#155eef' : row.scenario.startsWith('fedavg') ? '#da4f38' : '#137a68'} />)}</Bar></BarChart></ResponsiveContainer></div> : <EmptyState title="Saved comparisons unavailable" text="No valid metric rows were returned for the chart." />}
      </section>
      <section className="surface-card model-highlights"><SectionHeading title="A plain-language read" description="What these saved results show." />
        <div className="highlight-item"><span className="highlight-dot dot-blue" /><p><b>Centralized</b> had {highestAccuracy?.scenario === 'centralized' ? 'the strongest rounded accuracy' : 'rounded accuracy of ' + pct(central?.accuracy)}. This measures class matches, not whether traffic is moving well.</p></div>
        <div className="highlight-item"><span className="highlight-dot dot-teal" /><p><b>Location mean</b> had {bestError?.scenario === 'location_mean_baseline' ? 'the lowest MAE' : `MAE of ${fmt(location?.mae)}`}. It predicts from the average training label for each origin, so it is a useful reference—not a neural model.</p></div>
        <div className="highlight-item"><span className="highlight-dot dot-violet" /><p><b>FedAvg.</b> {fedavgComparison}</p></div>
        <div className="metric-notes"><Explain title="What does centralized learning mean?">One model is trained using the complete training split together. In this project it is compared against models trained on simulated client partitions.</Explain><Explain title="Why use several metrics?">Accuracy counts exact class matches. MAE measures average distance on the class scale. Macro-F1 gives equal weight to each congestion class. A method can rank differently on each measure.</Explain></div>
      </section>
    </div>
    <section className="surface-card comparison-card"><SectionHeading title="Compare every saved method" description="All values are averages across the available seed runs. No method is hidden." />
      <div className="table-scroll"><table><caption className="sr-only">Saved model scenario metrics</caption><thead><tr><th scope="col">Method</th><th scope="col">Mean absolute error</th><th scope="col">Rounded accuracy</th><th scope="col">Rounded macro-F1</th><th scope="col">Saved seeds</th></tr></thead><tbody>{summaries.filter(row => row.rows.length > 0).map(row => <tr key={row.scenario}><th scope="row">{scenarioName(row.scenario)}<small>{scenarioDescription(row.scenario)}</small></th><td>{fmt(row.mae)}</td><td>{pct(row.accuracy)}</td><td>{fmt(row.macroF1)}</td><td>{row.rows.map(item => item.seed).sort((a, b) => a - b).join(', ')}</td></tr>)}</tbody></table></div>
      <div className="metric-notes"><Explain title="What is MAE?">Mean absolute error: the average absolute difference between the raw predicted score and the true class. It is expressed in class levels—not minutes.</Explain><Explain title="What is rounded accuracy?">Both the true score and model score are rounded to the nearest class and clipped to classes 0–3. Accuracy is the share of records where the resulting classes match.</Explain><Explain title="What is macro-F1?">Macro-F1 averages each class’s precision and recall equally, including classes with fewer examples. Higher is better.</Explain></div>
    </section>
    <details className="technical-details"><summary>Experiment setup and technical details</summary><div className="detail-columns"><p><b>Evaluation split</b><br />{data?.split ?? 'Per-origin time-of-day holdout; no calendar dates are available.'}</p><p><b>Seeds in saved results</b><br />{seeds.length ? seeds.join(', ') : 'Not available'}</p><p><b>Central model</b><br />MLP with hidden layers 32 and 16; Huber loss; 60 epochs.</p><p><b>FedAvg setup</b><br />{fedavgClients !== undefined && fedavgRounds !== undefined ? `${fedavgClients} clients, ${fedavgRounds} rounds, one local epoch per round, sample-weighted averaging.` : 'Not available in the saved artifacts.'}</p><p><b>Input features</b><br />{data ? `${data.feature_count.toLocaleString()} encoded features: cyclic time plus origin IDs.` : 'Not available'}</p><p><b>Test examples</b><br />{data?.test_rows.toLocaleString() ?? 'Not available'}</p></div></details>
  </div>
}

function FederatedPage({ summaries }: { summaries: ScenarioSummary[] }) {
  const [status, setStatus] = useState<Awaited<ReturnType<typeof api.federated>> | null>(null)
  const [error, setError] = useState('')
  async function loadStatus() { setError(''); setStatus(null); try { setStatus(await api.federated()) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } }
  useEffect(() => { void loadStatus() }, [])
  const iid = summaries.find(row => row.scenario === 'fedavg_iid')
  const location = summaries.find(row => row.scenario === 'fedavg_location')

  return <div className="page-content">
    {error && <RequestFailure state={{ error }} onRetry={loadStatus} />}
    <section className="federated-intro surface-card"><div className="federated-icon"><Network size={24} /></div><div><span className="section-kicker">THE IDEA</span><h2>Everyone learns locally. The shared model learns from the updates.</h2><p>Think of four learners each practicing with a different part of the same traffic study. They send model updates to a server, which combines them and sends the improved model back.</p></div></section>
    <FederatedRouteDiagram clients={status?.clients} rounds={status?.rounds} />
    <div className="plain-note"><Info size={18} /><p>This project simulates client partitions locally. It does not involve independent companies or remote devices, and it does not demonstrate that model updates are private.</p></div>
    <div className="section-intro section-spaced"><div><span className="section-kicker">SAVED FEDAVG RESULTS</span><h2>Two ways to divide the examples</h2><p>Both partitions use the same evaluation split and the same model setup.</p></div>{status && <span className="data-badge"><CheckCircle2 size={16} />{status.available ? `${status.rounds} saved rounds` : 'No saved rounds'}</span>}</div>
    {!status && !error ? <LoadingCard label="Loading saved federated results…" /> : <div className="fed-result-grid">
      <FederatedResult title="IID partition" subtitle="Training examples are spread across clients in a balanced, shuffled way." result={iid} />
      <FederatedResult title="Location-based partition" subtitle="Each origin’s training examples stay together on one simulated client." result={location} />
    </div>}
    <details className="technical-details"><summary>What do “IID” and “location-based” mean?</summary><div className="detail-columns"><p><b>IID</b><br />Examples are distributed with a seeded random shuffle so clients get a similar mix of records.</p><p><b>Location-based</b><br />Whole origin groups are assigned to clients. The client data therefore reflects different locations.</p><p><b>One communication round</b><br />Each active client trains once, sends its model update, and receives the newly averaged model.</p><p><b>Saved experiment status</b><br />{status?.reason ?? 'Federated status is not available.'}</p></div></details>
  </div>
}

function FederatedRouteDiagram({ clients, rounds }: { clients?: number; rounds?: number | null }) {
  const count = clients ?? 4
  return <section className="federated-map" aria-labelledby="federated-map-title">
    <div className="federated-map-copy"><span className="section-kicker">THE LEARNING ROUTE</span><h2 id="federated-map-title">Four local routes meet at one shared model.</h2><p>Every simulated client trains on its assigned examples. FedAvg combines those updates by sample count, then returns the shared model for the next round.</p><ol className="route-legend"><li><b>1</b> Split historical training examples</li><li><b>2</b> Learn locally on each client</li><li><b>3</b> Aggregate with FedAvg</li><li><b>4</b> Repeat and evaluate</li></ol></div>
    <div className="federated-diagram" role="img" aria-label={`${count} simulated local clients send model updates to a central FedAvg aggregator. The averaged model returns to the clients for each of ${rounds ?? 'saved'} rounds.`}>
      <svg viewBox="0 0 640 390" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <defs><marker id="routeArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path fill="#155eef" d="M 0 0 L 10 5 L 0 10 z" /></marker><marker id="returnArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path fill="#da4f38" d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
        <path className="route-line route-in" d="M125 70 C250 70 240 150 310 177" markerEnd="url(#routeArrow)"/><path className="route-line route-in" d="M125 155 C210 155 235 177 310 187" markerEnd="url(#routeArrow)"/><path className="route-line route-in" d="M125 240 C210 240 235 215 310 198" markerEnd="url(#routeArrow)"/><path className="route-line route-in" d="M125 325 C250 325 240 230 310 208" markerEnd="url(#routeArrow)"/>
        <path className="route-line route-out" d="M335 177 C420 130 450 90 520 70" markerEnd="url(#returnArrow)"/><path className="route-line route-out" d="M335 187 C420 170 450 157 520 155" markerEnd="url(#returnArrow)"/><path className="route-line route-out" d="M335 198 C420 220 450 240 520 240" markerEnd="url(#returnArrow)"/><path className="route-line route-out" d="M335 208 C420 265 450 315 520 325" markerEnd="url(#returnArrow)"/>
        {[70,155,240,325].map((y, i) => <g key={y}><rect className="client-node" x="28" y={y - 27} width="103" height="54" rx="12"/><circle cx="49" cy={y} r="7"/><text x="64" y={y + 5}>Client {i + 1}</text><rect className="client-node client-return" x="520" y={y - 27} width="94" height="54" rx="12"/><text x="534" y={y + 5}>Client {i + 1}</text></g>)}
        <circle className="server-ring" cx="323" cy="192" r="63"/><circle className="server-node" cx="323" cy="192" r="48"/><text className="server-label" x="323" y="187">FedAvg</text><text className="server-subtitle" x="323" y="205">aggregator</text>
      </svg>
      <p><span>Model updates →</span><span>← Averaged model</span></p>
    </div>
    <p className="federated-caption">This is a diagram of the local simulation in this project, not a map of real devices or institutions.</p>
  </section>
}

function FederatedResult({ title, subtitle, result }: { title: string; subtitle: string; result?: ScenarioSummary }) {
  return <article className="surface-card fed-result-card"><div className="fed-title"><GitFork size={20} /><h3>{title}</h3></div><p>{subtitle}</p>{result?.rows.length ? <div className="fed-metrics"><div><span>Mean absolute error</span><b>{fmt(result.mae)}</b></div><div><span>Rounded accuracy</span><b>{pct(result.accuracy)}</b></div><div><span>Macro-F1</span><b>{fmt(result.macroF1)}</b></div></div> : <EmptyState title="No saved result for this partition" text="This scenario is not present in the result files." />}<span className="saved-seeds">Saved seeds: {result?.rows.length ? result.rows.map(row => row.seed).sort((a, b) => a - b).join(', ') : 'Not available'}</span></article>
}

function CommunicationPage() {
  const [estimate, setEstimate] = useState<Awaited<ReturnType<typeof api.communication>> | null>(null)
  const [error, setError] = useState('')
  async function loadEstimate() { setError(''); setEstimate(null); try { setEstimate(await api.communication()) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } }
  useEffect(() => { void loadEstimate() }, [])
  const bytes = estimate?.total_bytes_per_round
  const total = estimate?.total_bytes

  return <div className="page-content">
    {error && <RequestFailure state={{ error }} onRetry={loadEstimate} />}
    <section className="communication-explainer surface-card"><div className="communication-visual" aria-hidden="true"><div className="comm-node">Client</div><div className="comm-arrows"><span>Model update →</span><span>← Averaged model</span></div><div className="comm-node comm-server">Server</div></div><div><span className="section-kicker">WHY COMMUNICATION HAPPENS</span><h2>Models share updates, not traffic records.</h2><p>Each client sends a model update. The server combines those updates and sends the averaged model back. This project estimates how large those payloads would be.</p></div></section>
    {!estimate && !error ? <LoadingCard label="Loading saved communication estimate…" /> : estimate?.available ? <>
      <div className="estimate-grid">
        <EstimateCard label="One model payload" value={formatBytes(estimate.model_bytes)} description="Size of one full model update, based on its parameters." />
        <EstimateCard label="Estimated per round" value={formatBytes(bytes)} description="All client uploads plus all model deliveries for one round." tone="estimate-accent" />
        <EstimateCard label="Estimated full run" value={formatBytes(total)} description={`${estimate.rounds ?? 'Not available'} rounds for one saved ${scenarioName('fedavg_iid')} run.`} />
      </div>
      <section className="surface-card communication-breakdown"><SectionHeading title="What the estimate counts" description="Calculated from the saved model size and client count—not measured on a network." />
        <div className="exchange-row"><span className="exchange-icon"><ArrowRight size={18} /></span><div><strong>Clients send model updates</strong><p>One full model upload per client in each round.</p></div><b>{formatBytes(estimate.client_to_server_bytes_per_round)}</b></div>
        <div className="exchange-row"><span className="exchange-icon reverse"><ArrowRight size={18} /></span><div><strong>The server sends the averaged model back</strong><p>One full model delivery per client in each round.</p></div><b>{formatBytes(estimate.server_to_client_bytes_per_round)}</b></div>
        <div className="estimate-total"><span>Total estimated payload per round</span><strong>{formatBytes(bytes)}</strong></div>
      </section>
      <div className="plain-note"><Info size={18} /><p>These are payload-size estimates for four simulated clients. They exclude network headers, encryption, compression, retries, and other protocol costs. No real transfer was measured.</p></div>
      <details className="technical-details"><summary>Exact values and assumptions</summary><div className="detail-columns"><p><b>Model size</b><br />{exactBytes(estimate.model_bytes)} per model.</p><p><b>Client-to-server per round</b><br />{exactBytes(estimate.client_to_server_bytes_per_round)}.</p><p><b>Server-to-client per round</b><br />{exactBytes(estimate.server_to_client_bytes_per_round)}.</p><p><b>Total for one {estimate.rounds ?? 'saved'}-round run</b><br />{exactBytes(total)}.</p><p><b>Run represented</b><br />Seed 42, IID partition, {estimate.clients ?? 'not available'} clients.</p><p><b>Method</b><br />Full parameter upload and full model delivery, multiplied by client count and rounds.</p></div></details>
    </> : <EmptyState title="Communication estimate unavailable" text={estimate?.reason ?? 'No saved estimate was returned.'} />}
  </div>
}

function EstimateCard({ label, value, description, tone = '' }: { label: string; value: string; description: string; tone?: string }) {
  return <article className={`surface-card estimate-card ${tone}`}><span>{label}</span><strong>{value}</strong><p>{description}</p></article>
}

function ExperimentsPage({ data, summaries, runs, state, onRetry }: { data: Overview | null; summaries: ScenarioSummary[]; runs: MetricRow[]; state: RequestState; onRetry: () => void }) {
  const [filter, setFilter] = useState('all')
  const visible = summaries.filter(summary => summary.rows.length > 0 && (filter === 'all' || (filter === 'baseline' ? summary.scenario.endsWith('_baseline') : summary.scenario.startsWith(filter))))
  const seeds = [...new Set(runs.map(row => row.seed))].sort((a, b) => a - b)
  return <div className="page-content">
    <section className="experiment-intro surface-card"><BookOpen size={23} /><div><h2>Saved experiments answer different questions.</h2><p>Some models learn from all training examples, some keep examples in simulated client groups, and baselines provide simple points of comparison. Every method is evaluated on the held-aside test split.</p></div></section>
    {typeof state === 'object' ? <RequestFailure state={state} onRetry={onRetry} /> : state === 'loading' ? <LoadingCard label="Loading experiment history…" /> : <>
      <section className="surface-card experiment-summary"><SectionHeading title="Average results by method" description={`Mean across the saved seed runs${seeds.length ? ` (${seeds.join(', ')})` : ''}. Lower MAE is better; higher accuracy and macro-F1 are better.`} />
        <div className="filter-row" role="group" aria-label="Filter experiment methods">{[['all', 'All methods'], ['centralized', 'Centralized'], ['fedavg', 'FedAvg'], ['local_', 'Local-only'], ['baseline', 'Baselines']].map(([value, label]) => <button key={value} className={`filter-chip ${filter === value ? 'filter-active' : ''}`} onClick={() => setFilter(value)} aria-pressed={filter === value}>{label}</button>)}</div>
        <div className="table-scroll"><table><caption className="sr-only">Mean saved metrics by method</caption><thead><tr><th scope="col">Method</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">Rounded accuracy</th><th scope="col">Macro-F1</th></tr></thead><tbody>{visible.map(summary => <tr key={summary.scenario}><th scope="row">{scenarioName(summary.scenario)}<small>{scenarioDescription(summary.scenario)}</small></th><td>{fmt(summary.mae)}</td><td>{fmt(summary.rmse)}</td><td>{pct(summary.accuracy)}</td><td>{fmt(summary.macroF1)}</td></tr>)}</tbody></table></div>
      </section>
      <section className="surface-card run-details"><SectionHeading title="Per-seed results" description="Expand a method to see each saved seed. These files are existing experiment artifacts; this page does not run training." />
        {visible.map(summary => <details className="run-disclosure" key={summary.scenario}><summary><span>{scenarioName(summary.scenario)}</span><small>{summary.rows.length} saved seed results</small><ChevronDown size={17} /></summary><div className="table-scroll"><table><thead><tr><th scope="col">Seed</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">Rounded accuracy</th><th scope="col">Macro-F1</th></tr></thead><tbody>{summary.rows.slice().sort((a, b) => a.seed - b.seed).map(row => <tr key={row.seed}><th scope="row">{row.seed}</th><td>{fmt(row.mae)}</td><td>{fmt(row.rmse)}</td><td>{pct(row.accuracy_rounded)}</td><td>{fmt(row.macro_f1_rounded)}</td></tr>)}</tbody></table></div></details>)}
        {visible.length === 0 && <EmptyState title="No results match this filter" text="Choose another method group to see saved results." />}
      </section>
    </>}
    <div className="method-notes"><Explain title="How were the methods tested?">They use the same saved training/test split. The centralized model learns from all training records. Client methods divide training records into simulated groups. The test split is held aside for evaluation.</Explain><Explain title="What is a baseline?">A simple reference prediction helps show whether a more complex model adds value. The location-mean baseline predicts each origin’s average training congestion class; the training-mean baseline uses the overall training average.</Explain></div>
    <details className="technical-details"><summary>Study details and limitations</summary><div className="detail-columns"><p><b>Target classes</b><br />{data?.labels.join(' → ') ?? 'Very smooth → Smooth → Mild congestion → Heavy congestion'}</p><p><b>Data split</b><br />{data?.split ?? 'Time-of-day holdout by origin. No dates are available.'}</p><p><b>Unique saved seeds</b><br />{seeds.length ? seeds.join(', ') : 'Not available'}</p><p><b>Saved metric rows</b><br />{runs.length.toLocaleString()} unique method-and-seed rows after removing duplicate baseline rows.</p></div></details>
  </div>
}

function scenarioDescription(scenario: string) {
  const descriptions: Record<string, string> = {
    centralized: 'One model trained using all training records.',
    location_mean_baseline: 'Simple average label for each origin.',
    train_mean_baseline: 'One overall average label from training data.',
    fedavg_iid: 'FedAvg with shuffled, balanced client partitions.',
    fedavg_location: 'FedAvg with whole origins grouped by client.',
    local_iid_client_average: 'Separate local models; scores averaged across clients.',
    local_location_client_average: 'Separate local models on location groups.',
  }
  return descriptions[scenario] ?? 'Saved evaluation scenario.'
}

function numberValue(value: unknown): number | undefined { return typeof value === 'number' && Number.isFinite(value) ? value : undefined }
function exactBytes(value: number | undefined): string { return typeof value === 'number' ? `${value.toLocaleString()} bytes` : 'Not available' }
function formatBytes(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not available'
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1)} kB`
  return `${(value / 1_000_000).toFixed(2)} MB`
}
