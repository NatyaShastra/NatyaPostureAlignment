'use client'
import { useState, useCallback } from 'react'
import { analyseVideo, AnalysisResult } from '@/lib/api'
import UploadZone       from '@/components/UploadZone'
import AnalysingState   from '@/components/AnalysingState'
import ScoreCard        from '@/components/ScoreCard'
import FlaggedJoints    from '@/components/FlaggedJoints'
import CoachingFeedback from '@/components/CoachingFeedback'
import SkeletonOverlay  from '@/components/SkeletonOverlay'

type Stage = 'idle' | 'analysing' | 'results' | 'error'

export default function Home() {
  const [stage,    setStage]    = useState<Stage>('idle')
  const [filename, setFilename] = useState('')
  const [targetAdavu, setTargetAdavu] = useState<string>('Thattadavu')
  const [result,   setResult]   = useState<AnalysisResult | null>(null)
  const [error,    setError]    = useState('')

  const handleFile = useCallback(async (file: File) => {
    setFilename(file.name)
    setStage('analysing')
    setError('')
    setResult(null)

    try {
      const res = await analyseVideo(file, targetAdavu)
      setResult(res)
      setStage('results')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
      setStage('error')
    }
  }, [])

  function reset() {
    setStage('idle')
    setResult(null)
    setError('')
    setFilename('')
  }

  return (
    <div className="relative min-h-screen z-10">

      {/* Header */}
      <header className="border-b" style={{borderColor:'rgba(200,149,42,0.12)'}}>
        <div className="max-w-7xl mx-auto px-8 py-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Small decorative motif */}
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" className="opacity-80">
              {[0,45,90,135].map(deg => (
                <ellipse key={deg} cx="16" cy="16" rx="3" ry="11"
                  fill="none" stroke="#C8952A" strokeWidth="1.2"
                  transform={`rotate(${deg} 16 16)`} opacity="0.7"/>
              ))}
              <circle cx="16" cy="16" r="3" fill="#E8621A"/>
            </svg>
            <div>
              <h1 className="font-display text-xl leading-none" style={{color:'var(--ivory)'}}>
                Dance Coach <span style={{color:'var(--gold)'}}>AI</span>
              </h1>
              <p className="font-body text-xs mt-0.5" style={{color:'var(--ivory-dark)', opacity:0.5}}>
                Bharatanatyam Adavu Analysis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="font-body text-xs" style={{color:'var(--ivory-dark)', opacity:0.4}}>
              14 adavu classes · 88.7% accuracy · 9 joint angles
            </div>
            {stage === 'results' && (
              <button
                onClick={reset}
                className="font-body text-sm px-4 py-1.5 rounded-full transition-all"
                style={{
                  background:'rgba(232,98,26,0.12)',
                  color:'var(--saffron-light)',
                  border:'1px solid rgba(232,98,26,0.3)',
                }}
              >
                Analyse another
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-8 py-12">

        {/* ── IDLE ── */}
        {stage === 'idle' && (
          <div className="max-w-2xl mx-auto animate-fade-up">
            <div className="text-center mb-12">
              <p className="font-body text-lg mb-3" style={{color:'var(--gold)', opacity:0.8}}>
                ✦ Teacher Demo
              </p>
              <h2 className="font-display text-5xl leading-tight mb-4" style={{color:'var(--ivory)'}}>
                Adavu Analysis<br/>
                <span style={{color:'var(--saffron)'}}>in seconds</span>
              </h2>
              <p className="font-body text-lg leading-relaxed" style={{color:'var(--ivory-dark)', opacity:0.7}}>
                Upload a student's Bharatanatyam video and receive classification,
                joint-angle deviation analysis, a weighted score, and personalised
                coaching feedback.
              </p>
            </div>

            <div className="max-w-md mx-auto mb-6 text-left">
              <label className="font-body text-sm block mb-2" style={{color:'var(--ivory-dark)', opacity:0.8}}>
                Select Adavu to Grade:
              </label>
              <select 
                className="w-full bg-black border border-white/20 text-white rounded-lg px-4 py-3 outline-none focus:border-[#C8952A] transition-colors"
                value={targetAdavu}
                onChange={e => setTargetAdavu(e.target.value)}
              >
                <option value="Thattadavu">Thattadavu</option>
                <option value="Nattadavu">Nattadavu</option>
              </select>
            </div>

            <UploadZone onFile={handleFile} />

            {/* Feature list */}
            <div className="grid grid-cols-3 gap-4 mt-10">
              {[
                { icon: '◎', label: 'Adavu classification', sub: '14 classes · MLP classifier' },
                { icon: '⟡', label: 'Joint angle scoring',  sub: '9 angles · 1.5σ deviation' },
                { icon: '✦', label: 'LLM coaching',          sub: 'Groq · llama-3.3-70b' },
              ].map(f => (
                <div key={f.label} className="card rounded-xl p-4 text-center">
                  <div className="font-display text-2xl mb-2" style={{color:'var(--gold)'}}>{f.icon}</div>
                  <p className="font-display text-sm text-ivory mb-1">{f.label}</p>
                  <p className="font-body text-xs" style={{color:'var(--ivory-dark)', opacity:0.5}}>{f.sub}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── ANALYSING ── */}
        {stage === 'analysing' && (
          <AnalysingState filename={filename} />
        )}

        {/* ── ERROR ── */}
        {stage === 'error' && (
          <div className="max-w-xl mx-auto text-center py-20 animate-fade-up">
            <div className="text-5xl mb-6">✗</div>
            <h2 className="font-display text-2xl text-ivory mb-3">Analysis failed</h2>
            <p className="font-body text-base mb-8" style={{color:'var(--ivory-dark)', opacity:0.7}}>{error}</p>
            <button onClick={reset}
              className="font-body text-sm px-6 py-2 rounded-full"
              style={{background:'var(--saffron)', color:'var(--ivory)'}}>
              Try again
            </button>
          </div>
        )}

        {/* ── RESULTS ── */}
        {stage === 'results' && result && (
          <div className="animate-fade-up">

            {/* Results header */}
            <div className="flex items-center justify-between mb-8">
              <div>
                <p className="font-body text-sm" style={{color:'var(--gold)', opacity:0.7}}>Analysis complete</p>
                <h2 className="font-display text-3xl text-ivory">{filename}</h2>
              </div>
              <div className="font-body text-xs px-3 py-1 rounded-full"
                   style={{
                     background: result.passed ? 'rgba(76,175,80,0.12)' : 'rgba(139,26,47,0.15)',
                     color:      result.passed ? '#8BC34A'               : 'var(--crimson-light)',
                     border:     result.passed ? '1px solid rgba(76,175,80,0.3)' : '1px solid rgba(139,26,47,0.3)',
                   }}>
                {result.passed ? '✓ PASSED' : '✗ NOT PASSED'}
              </div>
            </div>

            {/* Three-column layout for teacher demo */}
            <div className="grid grid-cols-12 gap-6">

              {/* Left col: score + skeleton */}
              <div className="col-span-4 space-y-6">
                <ScoreCard result={result} />
                <SkeletonOverlay b64={result.overlay_image_b64} adavuClass={result.adavu_class} />
              </div>

              {/* Middle col: joint analysis */}
              <div className="col-span-4">
                <FlaggedJoints joints={result.flagged_joints} />
              </div>

              {/* Right col: coaching feedback */}
              <div className="col-span-4">
                <CoachingFeedback feedback={result.coaching_feedback} source={result.feedback_source} />
              </div>

            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t mt-20 py-6" style={{borderColor:'rgba(200,149,42,0.08)'}}>
        <div className="max-w-7xl mx-auto px-8 flex justify-between items-center">
          <p className="font-body text-xs" style={{color:'var(--ivory-dark)', opacity:0.3}}>
            Dance Coach AI · Bharatanatyam Adavu Analysis
          </p>
          <p className="font-body text-xs" style={{color:'var(--ivory-dark)', opacity:0.3}}>
            Backend · HuggingFace Spaces
          </p>
        </div>
      </footer>
    </div>
  )
}