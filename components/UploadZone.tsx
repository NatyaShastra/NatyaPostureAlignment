'use client'
import { useRef, useState, DragEvent, ChangeEvent } from 'react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

export default function UploadZone({ onFile, disabled }: Props) {
  const inputRef  = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) onFile(file)
  }

  return (
    <div
      className={`upload-zone rounded-2xl p-16 flex flex-col items-center justify-center cursor-pointer select-none
        ${dragging ? 'drag-over' : ''}
        ${disabled ? 'opacity-40 pointer-events-none' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      {/* Decorative lotus icon */}
      <svg className="w-16 h-16 mb-6 opacity-60" viewBox="0 0 64 64" fill="none">
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5"/>
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5" transform="rotate(30 32 40)"/>
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5" transform="rotate(60 32 40)"/>
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5" transform="rotate(90 32 40)"/>
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5" transform="rotate(120 32 40)"/>
        <ellipse cx="32" cy="40" rx="6" ry="14" fill="none" stroke="#C8952A" strokeWidth="1.5" transform="rotate(150 32 40)"/>
        <circle cx="32" cy="40" r="4" fill="#E8621A" opacity="0.8"/>
      </svg>

      <p className="font-display text-xl text-ivory mb-2">Drop a video here</p>
      <p className="font-body text-ink-light text-base" style={{color:'var(--ivory-dark)', opacity:0.7}}>
        or click to browse — MP4, MOV, AVI · max 50 MB
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
        className="hidden"
        onChange={handleChange}
      />
    </div>
  )
}