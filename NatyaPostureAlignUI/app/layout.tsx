import type { Metadata } from 'next'
import { Playfair_Display, Crimson_Pro } from 'next/font/google'
import './globals.css'

const playfair = Playfair_Display({
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
})

const crimson = Crimson_Pro({
  subsets: ['latin'],
  weight: ['300', '400', '600'],
  style: ['normal', 'italic'],
  variable: '--font-body',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Dance Coach AI — Bharatanatyam Adavu Analysis',
  description: 'AI-powered adavu coaching: classification, joint-angle scoring, and personalised feedback.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${playfair.variable} ${crimson.variable}`}>
      <body>{children}</body>
    </html>
  )
}