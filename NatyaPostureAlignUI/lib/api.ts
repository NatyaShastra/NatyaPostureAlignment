// lib/api.ts — typed API client for the Dance Coach backend

export interface FlaggedJoint {
  joint:         string
  measured:      number
  reference:     number
  deviation:     number
  deviation_deg: number
}

export interface JointComparisonRow {
  joint:         string
  left_student:  number
  left_master:   number
  left_diff:     number
  left_flagged:  boolean
  right_student: number
  right_master:  number
  right_diff:    number
  right_flagged: boolean
}

export interface Top5Anomaly {
  frame_index:        number
  video_frame:        number
  timestamp:          number
  anomaly_score:      number
  is_major_breach:    boolean
  student_image_b64:  string | null
  master_image_b64:   string | null
  master_video_name?: string
  master_frame_index?: number
  master_landmarks?:  number[][]
  comparison_table:   JointComparisonRow[]
}

export interface AnalysisResult {
  adavu_class:        string
  confidence:         number
  top_k_predictions:  [string, number][]
  overall_score:      number
  region_scores:      { legs: number; arms: number; torso?: number }
  passed:             boolean
  grade:              string
  grade_message:      string
  pass_threshold:     number
  needed_to_pass:     number
  flagged_joints:     FlaggedJoint[]
  coaching_feedback:  string
  feedback_source:    'llm' | 'template'
  overlay_image_b64:  string | null
  top_5_anomalies?:   Top5Anomaly[]
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export async function analyseVideo(file: File, category: string, targetClass: string): Promise<AnalysisResult> {
  const form = new FormData()
  form.append('video', file)
  form.append('category', category)
  form.append('target_class', targetClass)

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