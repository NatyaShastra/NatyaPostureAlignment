'use client'
import { useEffect, useState } from 'react'

const STEPS = [
  { label: 'Extracting pose landmarks…',     duration: 8000 },
  { label: 'Classifying adavu…',             duration: 4000 },
  { label: 'Computing joint angles…',        duration: 4000 },
  { label: 'Scoring your performance…',      duration: 3000 },
  { label: 'Generating coaching feedback…',  duration: 6000 },
  { label: 'Rendering skeleton overlay…',    duration: 3000 },
]

export default function AnalysingState({ filename }: { filename: string }) {
  const [stepIdx, setStepIdx] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (stepIdx >= STEPS.length - 1) return
    const t = setTimeout(() => setStepIdx(i => i + 1), STEPS[stepIdx].duration)
    return () => clearTimeout(t)
  }, [stepIdx])

  const progress = Math.min(((stepIdx + 1) / STEPS.length) * 100, 95)

  return (
    <div className="flex flex-col items-center justify-center py-20 animate-fade-up">

      {/* Spinning mandala */}
      <div className="relative w-24 h-24 mb-10">
        <svg className="absolute inset-0 animate-spin-slow" viewBox="0 0 96 96" style={{animationDuration:'12s'}}>
          {[0,30,60,90,120,150,180,210,240,270,300,330].map(deg => (
            <ellipse
              key={deg}
              cx="48" cy="48" rx="4" ry="20"
              fill="none" stroke="#C8952A" strokeWidth="1"
              opacity="0.5"
              transform={`rotate(${deg} 48 48)`}
            />
          ))}
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 rounded-full animate-pulse-slow" style={{background:'var(--saffron)', opacity:0.9}}/>
        </div>
      </div>

      <p className="font-display text-2xl text-ivory mb-1">Analysing</p>
      <p className="font-body text-base mb-10" style={{color:'var(--ivory-dark)', opacity:0.6}}>
        {filename}
      </p>

      {/* Progress bar */}
      <div className="w-80 h-1 rounded-full mb-4 overflow-hidden" style={{background:'rgba(200,149,42,0.15)'}}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{width:`${progress}%`, background:'linear-gradient(90deg, var(--saffron), var(--gold))'}}
        />
      </div>

      {/* Current step */}
      <p className="font-body text-sm" style={{color:'var(--gold-light)', opacity:0.8}}>
        {STEPS[stepIdx].label}
      </p>
      <p className="font-body text-xs mt-2" style={{color:'var(--ivory-dark)', opacity:0.4}}>
        {elapsed}s elapsed · free tier may take 30–60 s on cold start
      </p>
    </div>
  )
}