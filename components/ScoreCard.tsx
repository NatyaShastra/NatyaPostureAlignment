'use client'
import { AnalysisResult } from '@/lib/api'

interface Props { result: AnalysisResult }

const GRADE_COLORS: Record<string, string> = {
  A: '#4CAF50',
  B: '#8BC34A',
  C: '#C8952A',
  D: '#E8621A',
  F: '#8B1A2F',
}

function ScoreBar({ label, value, weight }: { label: string; value: number; weight: string }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-baseline mb-1">
        <span className="font-display text-sm capitalize" style={{color:'var(--ivory-dark)'}}>{label}</span>
        <span className="font-body text-xs" style={{color:'var(--gold)', opacity:0.7}}>{weight} weight</span>
        <span className="font-display text-lg" style={{color:'var(--ivory)'}}>{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{background:'rgba(200,149,42,0.12)'}}>
        <div
          className="score-bar-fill h-full rounded-full"
          style={{
            '--target-width': `${value}%`,
            background: value >= 70
              ? 'linear-gradient(90deg, #4CAF50, #8BC34A)'
              : value >= 50
              ? 'linear-gradient(90deg, var(--saffron), var(--gold))'
              : 'linear-gradient(90deg, var(--crimson), #E8621A)',
          } as React.CSSProperties}
        />
      </div>
    </div>
  )
}

export default function ScoreCard({ result }: Props) {
  const gradeColor = GRADE_COLORS[result.grade] ?? '#C8952A'

  return (
    <div className="card rounded-2xl p-8 animate-fade-up">

      {/* Header row */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <p className="font-body text-sm mb-1" style={{color:'var(--gold)', opacity:0.8}}>Detected adavu</p>
          <h2 className="font-display text-3xl text-ivory">{result.adavu_class}</h2>
          <p className="font-body text-sm mt-1" style={{color:'var(--ivory-dark)', opacity:0.6}}>
            {(result.confidence * 100).toFixed(1)}% confidence
          </p>
        </div>

        {/* Grade badge */}
        <div className="flex flex-col items-center">
          <div
            className="grade-badge w-16 h-16 rounded-full flex items-center justify-center border-2"
            style={{color: gradeColor, borderColor: gradeColor, background:`${gradeColor}18`}}
          >
            {result.grade}
          </div>
          <p className="font-body text-xs mt-1" style={{color:'var(--ivory-dark)', opacity:0.5}}>
            {result.passed ? '✓ Passed' : '✗ Not passed'}
          </p>
        </div>
      </div>

      {/* Overall score */}
      <div className="mb-6 p-4 rounded-xl" style={{background:'rgba(200,149,42,0.07)', border:'1px solid rgba(200,149,42,0.15)'}}>
        <div className="flex justify-between items-center mb-2">
          <span className="font-display text-base" style={{color:'var(--ivory-dark)'}}>Overall Score</span>
          <span className="font-display text-4xl" style={{color: gradeColor}}>{result.overall_score.toFixed(1)}</span>
        </div>
        <div className="h-3 rounded-full overflow-hidden" style={{background:'rgba(200,149,42,0.12)'}}>
          <div
            className="score-bar-fill h-full rounded-full"
            style={{
              '--target-width': `${result.overall_score}%`,
              background:`linear-gradient(90deg, ${gradeColor}88, ${gradeColor})`,
            } as React.CSSProperties}
          />
        </div>
        <p className="font-body text-xs mt-2" style={{color:'var(--ivory-dark)', opacity:0.5}}>
          {result.grade_message}
        </p>
      </div>

      {/* Region breakdown */}
      <div className="divider-kolam mb-6">
        <span className="text-xs font-body px-2" style={{color:'var(--gold)'}}>By region</span>
      </div>

      <ScoreBar label="Legs"  value={result.region_scores.legs}  weight="50%" />
      <ScoreBar label="Arms"  value={result.region_scores.arms}  weight="30%" />
      <ScoreBar label="Torso" value={result.region_scores.torso} weight="20%" />

      {/* Pass threshold note */}
      {!result.passed && (
        <p className="font-body text-sm mt-4 px-3 py-2 rounded-lg"
           style={{color:'var(--saffron-light)', background:'rgba(232,98,26,0.08)', border:'1px solid rgba(232,98,26,0.2)'}}>
          Need {result.needed_to_pass.toFixed(1)}% more to pass (threshold: {result.pass_threshold}%)
        </p>
      )}

      {/* Alternate candidates */}
      {result.top_k_predictions.length > 1 && (
        <div className="mt-6">
          <p className="font-body text-xs mb-2" style={{color:'var(--ivory-dark)', opacity:0.5}}>Other candidates</p>
          <div className="space-y-1">
            {result.top_k_predictions.slice(1).map(([name, conf]) => (
              <div key={name} className="flex justify-between font-body text-sm" style={{color:'var(--ivory-dark)', opacity:0.6}}>
                <span>{name}</span>
                <span>{(conf * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}