// lib/api.ts — typed API client for the Dance Coach backend

export interface FlaggedJoint {
  joint:         string
  measured:      number
  reference:     number
  deviation:     number
  deviation_deg: number
}

export interface AnalysisResult {
  adavu_class:        string
  confidence:         number
  top_k_predictions:  [string, number][]
  overall_score:      number
  region_scores:      { legs: number; arms: number; torso: number }
  passed:             boolean
  grade:              string
  grade_message:      string
  pass_threshold:     number
  needed_to_pass:     number
  flagged_joints:     FlaggedJoint[]
  coaching_feedback:  string
  feedback_source:    'llm' | 'template'
  overlay_image_b64:  string | null
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'https://theusefulnerd-dance-coach-ai.hf.space'

export async function analyseVideo(file: File): Promise<AnalysisResult> {
  const form = new FormData()
  form.append('video', file)

  const res = await fetch(`${API_URL}/analyse`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Server error ${res.status}`)
  }

  return res.json()
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store' })
    return res.ok
  } catch {
    return false
  }
}