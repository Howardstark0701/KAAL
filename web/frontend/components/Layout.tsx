import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

interface LayoutProps { children: React.ReactNode; }

const NAV_LINKS = [
  { href: '/',        label: 'Home' },
  { href: '/audit',   label: 'Audit' },
  { href: '/patch',   label: 'Patch' },
  { href: '/compare', label: 'Compare' },
];

export default function Layout({ children }: LayoutProps) {
  const { pathname } = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-gray-100 flex flex-col">
      {/* Nav */}
      <header className="sticky top-0 z-50 border-b border-[#222] bg-[#0A0A0A]/95 backdrop-blur">
        <nav className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-bold text-white tracking-widest text-lg">
            <span className="text-[#CC0000] font-mono">KAAL</span>
          </Link>

          {/* Desktop links */}
          <ul className="hidden md:flex items-center gap-1" role="menubar">
            {NAV_LINKS.map(({ href, label }) => {
              const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
              return (
                <li key={href} role="none">
                  <Link
                    href={href}
                    role="menuitem"
                    className={`px-3 py-1.5 rounded text-sm font-medium transition-colors
                      ${active
                        ? 'bg-[#CC0000]/15 text-[#CC0000]'
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }`}
                  >
                    {label}
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 rounded text-gray-400 hover:text-white focus-visible:ring-2 focus-visible:ring-[#CC0000]"
            aria-label="Toggle navigation menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span className="block w-5 h-0.5 bg-current mb-1" />
            <span className="block w-5 h-0.5 bg-current mb-1" />
            <span className="block w-5 h-0.5 bg-current" />
          </button>
        </nav>

        {/* Mobile menu */}
        {menuOpen && (
          <div className="md:hidden border-t border-[#222] bg-[#111] px-4 py-2">
            {NAV_LINKS.map(({ href, label }) => {
              const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setMenuOpen(false)}
                  className={`block px-3 py-2 rounded text-sm font-medium mb-1
                    ${active ? 'text-[#CC0000] bg-[#CC0000]/10' : 'text-gray-400 hover:text-white'}`}
                >
                  {label}
                </Link>
              );
            })}
          </div>
        )}
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-[#222] text-center text-xs text-gray-600 py-4 flex items-center justify-center gap-4">
        <span>KAAL v1.0.0 — Adversarial Robustness Auditing Tool</span>
        <span>·</span>
        <a href="https://github.com/Howardstark0701/KAAL" target="_blank" rel="noreferrer" className="hover:text-gray-400 transition-colors">GitHub</a>
        <span>·</span>
        <span>MIT License</span>
      </footer>
    </div>
  );
}
