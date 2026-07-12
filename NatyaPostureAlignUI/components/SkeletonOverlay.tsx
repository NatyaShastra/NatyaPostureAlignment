'use client'
import { useState } from 'react'

interface Props { b64: string | null; adavuClass: string }

export default function SkeletonOverlay({ b64, adavuClass }: Props) {
  const [enlarged, setEnlarged] = useState(false)

  if (!b64) return null

  const src = `data:image/jpeg;base64,${b64}`

  return (
    <>
      <div className="card rounded-2xl p-8 animate-fade-up delay-400">
        <h3 className="font-display text-xl text-ivory mb-4">Skeleton Overlay</h3>
        <p className="font-body text-sm mb-4" style={{color:'var(--ivory-dark)', opacity:0.5}}>
          <span style={{color:'#8BC34A'}}>●</span> Within range &nbsp;&nbsp;
          <span style={{color:'#E84032'}}>●</span> Needs correction
        </p>
        <img
          src={src}
          alt={`Skeleton overlay for ${adavuClass}`}
          className="w-full rounded-xl cursor-zoom-in object-contain"
          style={{maxHeight:'360px', border:'1px solid rgba(200,149,42,0.15)'}}
          onClick={() => setEnlarged(true)}
        />
        <p className="font-body text-xs mt-2 text-center" style={{color:'var(--ivory-dark)', opacity:0.4}}>
          Click to enlarge
        </p>
      </div>

      {/* Lightbox */}
      {enlarged && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-8"
          style={{background:'rgba(28,17,8,0.92)', backdropFilter:'blur(8px)'}}
          onClick={() => setEnlarged(false)}
        >
          <img
            src={src}
            alt={`Skeleton overlay for ${adavuClass}`}
            className="max-w-full max-h-full rounded-2xl"
            style={{border:'1px solid rgba(200,149,42,0.3)'}}
          />
          <button
            className="absolute top-6 right-6 font-body text-sm px-3 py-1 rounded-full"
            style={{background:'rgba(200,149,42,0.2)', color:'var(--gold-light)', border:'1px solid rgba(200,149,42,0.3)'}}
          >
            Close
          </button>
        </div>
      )}
    </>
  )
}