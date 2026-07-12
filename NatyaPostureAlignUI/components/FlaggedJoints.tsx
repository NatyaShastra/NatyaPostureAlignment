'use client'
import { FlaggedJoint } from '@/lib/api'

interface Props { joints: FlaggedJoint[] }

const BODY_REGION: Record<string, string> = {
  left_knee: 'Legs', right_knee: 'Legs',
  left_hip:  'Legs', right_hip:  'Legs',
  left_elbow: 'Arms', right_elbow: 'Arms',
  left_shoulder: 'Arms', right_shoulder: 'Arms',
  spine_lean: 'Torso',
}

const REGION_COLOR: Record<string, string> = {
  Legs:  '#E8621A',
  Arms:  '#C8952A',
  Torso: '#8B1A2F',
}

function SigmaBar({ deviation }: { deviation: number }) {
  const capped  = Math.min(deviation, 4)
  const pct     = (capped / 4) * 100
  const color   = deviation > 3 ? '#8B1A2F' : deviation > 2 ? '#E8621A' : '#C8952A'

  return (
    <div className="flex items-center gap-2 mt-1">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{background:'rgba(200,149,42,0.12)'}}>
        <div className="h-full rounded-full transition-all duration-700"
             style={{width:`${pct}%`, background: color}}/>
      </div>
      <span className="font-body text-xs" style={{color, minWidth:'3.5rem', textAlign:'right'}}>
        {deviation.toFixed(2)}σ
      </span>
    </div>
  )
}

export default function FlaggedJoints({ joints }: Props) {
  if (joints.length === 0) {
    return (
      <div className="card rounded-2xl p-8 animate-fade-up delay-200">
        <h3 className="font-display text-xl text-ivory mb-4">Joint Analysis</h3>
        <div className="flex flex-col items-center py-8 opacity-60">
          <svg className="w-12 h-12 mb-3" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="20" stroke="#4CAF50" strokeWidth="2"/>
            <path d="M14 24l7 7 13-13" stroke="#4CAF50" strokeWidth="2.5" strokeLinecap="round"/>
          </svg>
          <p className="font-body text-base" style={{color:'var(--ivory-dark)'}}>
            No major joint deviations detected
          </p>
          <p className="font-body text-sm" style={{color:'var(--ivory-dark)', opacity:0.5}}>
            All 9 angles within 1.5σ of reference
          </p>
        </div>
      </div>
    )
  }

  const sorted = [...joints].sort((a, b) => b.deviation - a.deviation)

  return (
    <div className="card rounded-2xl p-8 animate-fade-up delay-200">
      <div className="flex items-baseline justify-between mb-6">
        <h3 className="font-display text-xl text-ivory">Joint Analysis</h3>
        <span className="font-body text-sm px-2 py-0.5 rounded-full"
              style={{background:'rgba(232,98,26,0.15)', color:'var(--saffron-light)', border:'1px solid rgba(232,98,26,0.3)'}}>
          {joints.length} flagged
        </span>
      </div>

      <div className="space-y-4">
        {sorted.map((j, i) => {
          const region    = BODY_REGION[j.joint] ?? 'Other'
          const color     = REGION_COLOR[region]  ?? '#C8952A'
          const label     = j.joint.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
          const direction = j.measured > j.reference ? 'too open' : 'too closed'

          return (
            <div key={i} className="p-4 rounded-xl"
                 style={{background:'rgba(61,43,20,0.6)', border:`1px solid ${color}28`}}>
              <div className="flex items-start justify-between">
                <div>
                  <span className="font-display text-base text-ivory">{label}</span>
                  <span className="ml-2 font-body text-xs px-1.5 py-0.5 rounded"
                        style={{background:`${color}22`, color}}>
                    {region}
                  </span>
                </div>
                <div className="text-right">
                  <span className="font-body text-sm" style={{color:'var(--ivory-dark)'}}>
                    {j.measured.toFixed(1)}°
                  </span>
                  <span className="font-body text-xs ml-1" style={{color:'var(--ivory-dark)', opacity:0.4}}>
                    vs {j.reference.toFixed(1)}° ref
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between mt-1">
                <span className="font-body text-xs italic" style={{color:'var(--ivory-dark)', opacity:0.5}}>
                  {j.deviation_deg.toFixed(1)}° {direction}
                </span>
              </div>

              <SigmaBar deviation={j.deviation} />
            </div>
          )
        })}
      </div>
    </div>
  )
}