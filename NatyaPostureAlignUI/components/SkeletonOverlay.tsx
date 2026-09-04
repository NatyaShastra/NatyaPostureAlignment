'use client'
import { useState } from 'react'
import { Top5Anomaly } from '@/lib/api'

interface Props {
  b64: string | null
  adavuClass: string
  top5?: Top5Anomaly[]
}

const POSE_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left Arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right Arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left Leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right Leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
]

export default function SkeletonOverlay({ b64, adavuClass, top5 }: Props) {
  const [selectedTab, setSelectedTab] = useState(0)
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)
  const [lightboxTitle, setLightboxTitle] = useState<string>('')

  // If no advanced DTW Top 5 payload is available, fall back to basic single overlay
  if (!top5 || top5.length === 0) {
    if (!b64) return null
    const src = `data:image/jpeg;base64,${b64}`
    return (
      <>
        <div className="card rounded-2xl p-8 animate-fade-up delay-400">
          <h3 className="font-display text-xl text-ivory mb-4">Skeleton Overlay</h3>
          <p className="font-body text-sm mb-4" style={{ color: 'var(--ivory-dark)', opacity: 0.5 }}>
            <span style={{ color: '#8BC34A' }}>●</span> Within range &nbsp;&nbsp;
            <span style={{ color: '#E84032' }}>●</span> Needs correction
          </p>
          <img
            src={src}
            alt={`Skeleton overlay for ${adavuClass}`}
            className="w-full rounded-xl cursor-zoom-in object-contain"
            style={{ maxHeight: '360px', border: '1px solid rgba(200,149,42,0.15)' }}
            onClick={() => { setLightboxSrc(src); setLightboxTitle(`Skeleton overlay for ${adavuClass}`) }}
          />
          <p className="font-body text-xs mt-2 text-center" style={{ color: 'var(--ivory-dark)', opacity: 0.4 }}>
            Click to enlarge
          </p>
        </div>
        {lightboxSrc && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-8"
            style={{ background: 'rgba(28,17,8,0.92)', backdropFilter: 'blur(8px)' }}
            onClick={() => setLightboxSrc(null)}
          >
            <img
              src={lightboxSrc}
              alt={lightboxTitle}
              className="max-w-full max-h-full rounded-2xl"
              style={{ border: '1px solid rgba(200,149,42,0.3)' }}
            />
            <button
              className="absolute top-6 right-6 font-body text-sm px-3 py-1 rounded-full"
              style={{ background: 'rgba(200,149,42,0.2)', color: 'var(--gold-light)', border: '1px solid rgba(200,149,42,0.3)' }}
            >
              Close
            </button>
          </div>
        )}
      </>
    )
  }

  const current = top5[selectedTab] || top5[0]
  const studentSrc = current.student_image_b64 ? `data:image/jpeg;base64,${current.student_image_b64}` : (b64 ? `data:image/jpeg;base64,${b64}` : null)
  
  const bunnyCdnUrl = process.env.NEXT_PUBLIC_BUNNY_CDN_URL || 'https://natyamaster.b-cdn.net'
  const paddedMasterIndex = String(current.master_frame_index ?? 0).padStart(3, '0')
  const masterVideoFolder = current.master_video_name ?? adavuClass
  const masterSrc = current.master_frame_index !== undefined && current.master_video_name
    ? `${bunnyCdnUrl}/${masterVideoFolder}/frame_${paddedMasterIndex}.jpg`
    : (current.master_image_b64 ? `data:image/jpeg;base64,${current.master_image_b64}` : null)

  return (
    <>
      <div className="card rounded-2xl p-8 animate-fade-up delay-400">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-body px-2 py-0.5 rounded uppercase font-bold" style={{ background: 'rgba(200,149,42,0.15)', color: 'var(--gold)' }}>
                DTW Temporal Alignment
              </span>
            </div>
            <h3 className="font-display text-2xl text-ivory">
              Top 5 Temporal Form Breakdowns
            </h3>
            <p className="font-body text-sm mt-1" style={{ color: 'var(--ivory-dark)', opacity: 0.6 }}>
              Side-by-side comparison against the pristine Gold-Standard Master trajectory.
            </p>
          </div>
        </div>

        {/* Tab Bar / Timeline Selection */}
        <div className="flex flex-wrap gap-2 mb-8 p-1.5 rounded-xl bg-black/20 border border-[rgba(200,149,42,0.15)]">
          {top5.map((item, idx) => {
            const isSelected = selectedTab === idx
            return (
              <button
                key={idx}
                onClick={() => setSelectedTab(idx)}
                className={`flex-1 min-w-[140px] px-3 py-2.5 rounded-lg text-left transition-all duration-200 flex flex-col justify-between border ${
                  isSelected
                    ? 'bg-[rgba(200,149,42,0.2)] border-[var(--gold)] shadow-lg'
                    : 'bg-transparent border-transparent hover:bg-white/5 opacity-70 hover:opacity-100'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-display text-sm font-bold text-ivory">
                    #{idx + 1} Frame {item.video_frame}
                  </span>
                  <span className="font-body text-xs text-ivory/60">
                    ⏱ {item.timestamp}s
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${item.is_major_breach ? 'bg-[#E84032] animate-pulse' : 'bg-[#E8621A]'}`} />
                  <span className={`font-body text-xs font-semibold ${item.is_major_breach ? 'text-[#E84032]' : 'text-[#E8621A]'}`}>
                    {item.anomaly_score.toFixed(0)}% {item.is_major_breach ? 'Major Breach' : 'Deviation'}
                  </span>
                </div>
              </button>
            )
          })}
        </div>

        {/* Side-by-Side Dual Image Showcase */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Master Reference Skeleton */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="font-display text-base text-ivory flex items-center gap-2">
                <span className="text-[#8BC34A]">●</span> Master Reference Skeleton
              </span>
              <span className="font-body text-xs text-ivory/50">Gold Standard</span>
            </div>
            <div className="relative rounded-xl overflow-hidden bg-[#1a1512] border border-[rgba(200,149,42,0.2)] aspect-square flex items-center justify-center group">
              {masterSrc ? (
                <>
                  <img
                    src={masterSrc}
                    alt="Master Reference"
                    className="w-full h-full object-contain cursor-zoom-in transition-transform duration-300 group-hover:scale-[1.02]"
                    onClick={() => { setLightboxSrc(masterSrc); setLightboxTitle('Master Reference (Gold Standard)') }}
                  />
                  {current.master_landmarks && current.master_landmarks.length > 0 && (
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      viewBox="0 0 100 100"
                      preserveAspectRatio="xMidYMid meet"
                    >
                      {POSE_CONNECTIONS.map(([i, j], idx) => {
                        const pt1 = current.master_landmarks![i]
                        const pt2 = current.master_landmarks![j]
                        if (!pt1 || !pt2 || (pt1[2] !== undefined && pt1[2] < 0.2) || (pt2[2] !== undefined && pt2[2] < 0.2)) {
                          return null
                        }
                        return (
                          <line
                            key={idx}
                            x1={pt1[0] * 100}
                            y1={pt1[1] * 100}
                            x2={pt2[0] * 100}
                            y2={pt2[1] * 100}
                            stroke="#8BC34A"
                            strokeWidth="1.2"
                            strokeLinecap="round"
                            opacity="0.9"
                          />
                        )
                      })}
                      {current.master_landmarks.map((lm, idx) => {
                        if (!lm || (lm[2] !== undefined && lm[2] < 0.2)) return null
                        return (
                          <circle
                            key={idx}
                            cx={lm[0] * 100}
                            cy={lm[1] * 100}
                            r="1"
                            fill="#8BC34A"
                            stroke="#FFFFFF"
                            strokeWidth="0.3"
                          />
                        )
                      })}
                    </svg>
                  )}
                  <div className="absolute bottom-3 right-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] text-ivory/80 pointer-events-none">
                    Click to enlarge ↗
                  </div>
                </>
              ) : (
                <div className="text-center p-6 text-ivory/40 font-body text-sm">
                  Reference pose rendering...
                </div>
              )}
            </div>
          </div>

          {/* Student Execution */}
          <div className="flex flex-col">
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="font-display text-base text-ivory flex items-center gap-2">
                <span className="text-[#E84032]">●</span> Student Execution
              </span>
              <span className="font-body text-xs text-ivory/50">Frame #{current.video_frame} ({current.timestamp}s)</span>
            </div>
            <div className="relative rounded-xl overflow-hidden bg-[#1a1512] border border-[rgba(200,149,42,0.2)] aspect-square flex items-center justify-center group">
              {studentSrc ? (
                <>
                  <img
                    src={studentSrc}
                    alt="Student Execution"
                    className="w-full h-full object-contain cursor-zoom-in transition-transform duration-300 group-hover:scale-[1.02]"
                    onClick={() => { setLightboxSrc(studentSrc); setLightboxTitle(`Student Execution (Frame #${current.video_frame}, ⏱ ${current.timestamp}s)`) }}
                  />
                  <div className="absolute bottom-3 right-3 bg-black/60 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] text-ivory/80 pointer-events-none">
                    Click to enlarge ↗
                  </div>
                </>
              ) : (
                <div className="text-center p-6 text-ivory/40 font-body text-sm">
                  No image available
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Left & Right Joint Comparison Table */}
        <div className="bg-[#140e0a]/80 rounded-xl border border-[rgba(200,149,42,0.15)] overflow-hidden">
          <div className="px-5 py-3.5 bg-black/30 border-b border-[rgba(200,149,42,0.15)] flex items-center justify-between">
            <h4 className="font-display text-base text-ivory">
              Bilateral Joint Angle Comparison (Frame #{current.video_frame})
            </h4>
            <span className="font-body text-xs text-ivory/60">
              Red values indicate deviation &gt;1.5σ from master
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[rgba(200,149,42,0.1)] text-xs font-display text-ivory/60 uppercase tracking-wider bg-black/20">
                  <th className="py-3 px-4">Joint</th>
                  <th className="py-3 px-4 text-center">Left (Student)</th>
                  <th className="py-3 px-4 text-center">Left (Master)</th>
                  <th className="py-3 px-4 text-center">Left Diff</th>
                  <th className="py-3 px-4 text-center border-l border-[rgba(200,149,42,0.1)]">Right (Student)</th>
                  <th className="py-3 px-4 text-center">Right (Master)</th>
                  <th className="py-3 px-4 text-center">Right Diff</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[rgba(200,149,42,0.06)] font-body text-sm">
                {current.comparison_table.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 px-4 font-display font-medium text-ivory">{row.joint}</td>
                    
                    {/* Left Side */}
                    <td className="py-3 px-4 text-center text-ivory/80">{row.left_student}°</td>
                    <td className="py-3 px-4 text-center text-ivory/50">{row.left_master}°</td>
                    <td className={`py-3 px-4 text-center font-bold ${row.left_flagged ? 'text-[#E84032] bg-[#E84032]/10 rounded' : 'text-[#8BC34A]'}`}>
                      {row.left_diff}° {row.left_flagged ? '⚠️' : ''}
                    </td>

                    {/* Right Side */}
                    <td className="py-3 px-4 text-center text-ivory/80 border-l border-[rgba(200,149,42,0.1)]">{row.right_student}°</td>
                    <td className="py-3 px-4 text-center text-ivory/50">{row.right_master}°</td>
                    <td className={`py-3 px-4 text-center font-bold ${row.right_flagged ? 'text-[#E84032] bg-[#E84032]/10 rounded' : 'text-[#8BC34A]'}`}>
                      {row.right_diff}° {row.right_flagged ? '⚠️' : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightboxSrc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/95 backdrop-blur-md animate-fade-in"
          onClick={() => setLightboxSrc(null)}
        >
          <div className="relative max-w-4xl w-full max-h-[90vh] flex flex-col items-center justify-center">
            <div className="w-full flex items-center justify-between mb-3 px-2">
              <span className="font-display text-lg text-ivory">{lightboxTitle}</span>
              <button
                className="font-body text-sm px-4 py-1.5 rounded-full bg-[rgba(200,149,42,0.2)] text-[var(--gold)] hover:bg-[rgba(200,149,42,0.3)] transition-colors border border-[rgba(200,149,42,0.3)]"
                onClick={() => setLightboxSrc(null)}
              >
                Close ✕
              </button>
            </div>
            <img
              src={lightboxSrc}
              alt={lightboxTitle}
              className="max-w-full max-h-[80vh] rounded-2xl object-contain border border-[rgba(200,149,42,0.3)] shadow-2xl bg-black"
            />
          </div>
        </div>
      )}
    </>
  )
}