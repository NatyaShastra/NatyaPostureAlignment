'use client'

interface Props {
  feedback: string
  source:   'llm' | 'template'
}

function parseSection(text: string, header: string): string | null {
  // Looks for "1. WHAT YOU DID WELL", "2. CORRECTIONS NEEDED", "3. NEXT PRACTICE FOCUS"
  const regex = new RegExp(`${header}[\\s\\S]*?(?=\\n\\n\\d\\.|$)`, 'i')
  const match = text.match(regex)
  if (!match) return null
  return match[0].replace(new RegExp(header, 'i'), '').trim()
}

export default function CoachingFeedback({ feedback, source }: Props) {
  // Try to parse structured LLM response into three sections
  const wellDone   = parseSection(feedback, '1\\.\\s*WHAT YOU DID WELL')
                  ?? parseSection(feedback, 'WHAT YOU DID WELL')
  const corrections= parseSection(feedback, '2\\.\\s*CORRECTIONS NEEDED')
                  ?? parseSection(feedback, 'CORRECTIONS NEEDED')
  const practice   = parseSection(feedback, '3\\.\\s*NEXT PRACTICE FOCUS')
                  ?? parseSection(feedback, 'NEXT PRACTICE FOCUS')

  const isStructured = wellDone && corrections && practice

  return (
    <div className="card rounded-2xl p-8 animate-fade-up delay-300">
      <div className="flex items-baseline justify-between mb-6">
        <h3 className="font-display text-xl text-ivory">Coaching Feedback</h3>
        <span className="font-body text-xs px-2 py-0.5 rounded-full"
              style={{
                background: source === 'llm' ? 'rgba(200,149,42,0.15)' : 'rgba(61,43,20,0.6)',
                color:      source === 'llm' ? 'var(--gold-light)'      : 'var(--ivory-dark)',
                border:     source === 'llm' ? '1px solid rgba(200,149,42,0.3)' : '1px solid rgba(200,149,42,0.1)',
                opacity:    source === 'llm' ? 1 : 0.6,
              }}>
          {source === 'llm' ? '✦ AI Generated' : 'Template'}
        </span>
      </div>

      {isStructured ? (
        <div className="space-y-5">
          {/* What you did well */}
          <div className="p-4 rounded-xl" style={{background:'rgba(76,175,80,0.07)', border:'1px solid rgba(76,175,80,0.2)'}}>
            <p className="font-display text-sm mb-2" style={{color:'#8BC34A'}}>What you did well</p>
            <p className="font-body text-base leading-relaxed" style={{color:'var(--ivory-dark)'}}>{wellDone}</p>
          </div>

          {/* Corrections */}
          <div className="p-4 rounded-xl" style={{background:'rgba(232,98,26,0.06)', border:'1px solid rgba(232,98,26,0.2)'}}>
            <p className="font-display text-sm mb-2" style={{color:'var(--saffron-light)'}}>Corrections needed</p>
            <p className="font-body text-base leading-relaxed whitespace-pre-line" style={{color:'var(--ivory-dark)'}}>
              {corrections}
            </p>
          </div>

          {/* Next practice */}
          <div className="p-4 rounded-xl" style={{background:'rgba(200,149,42,0.07)', border:'1px solid rgba(200,149,42,0.2)'}}>
            <p className="font-display text-sm mb-2" style={{color:'var(--gold-light)'}}>Next practice focus</p>
            <p className="font-body text-base leading-relaxed" style={{color:'var(--ivory-dark)'}}>{practice}</p>
          </div>
        </div>
      ) : (
        // Fallback: render raw text
        <p className="font-body text-base leading-relaxed whitespace-pre-line" style={{color:'var(--ivory-dark)'}}>
          {feedback}
        </p>
      )}
    </div>
  )
}