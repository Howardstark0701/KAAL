import Head from 'next/head';
import Link from 'next/link';

const ATTACKS = [
  {
    name: 'FGSM',
    sub: 'Fast Gradient Sign Method',
    desc: 'Single-step gradient attack. Produces imperceptible perturbations that reliably cause misclassification.',
    icon: '⚡',
  },
  {
    name: 'PGD',
    sub: 'Projected Gradient Descent',
    desc: 'Iterative FGSM with L∞ projection. The gold standard adversarial attack — stronger and harder to defend against.',
    icon: '🔁',
  },
  {
    name: 'Patch',
    sub: 'Adversarial Patch',
    desc: 'A printable sticker that causes misclassification regardless of position on the image.',
    icon: '🩹',
  },
  {
    name: 'Physical',
    sub: 'Robustness Simulator',
    desc: 'Tests whether adversarial examples survive 26 real-world transforms: JPEG, blur, rotation, lighting, noise.',
    icon: '🌍',
  },
];

const KVS_TIERS = [
  { range: '0.0 – 2.0',  label: 'Robust',       color: '#4ADE80' },
  { range: '2.1 – 4.0',  label: 'Low Risk',      color: '#A3E635' },
  { range: '4.1 – 6.0',  label: 'Medium Risk',   color: '#FACC15' },
  { range: '6.1 – 8.0',  label: 'High Risk',     color: '#FB923C' },
  { range: '8.1 – 9.5',  label: 'Critical',      color: '#CC0000' },
  { range: '9.6 – 10.0', label: 'Catastrophic',  color: '#7F0000' },
];

export default function HomePage() {
  return (
    <>
      <Head><title>KAAL — Adversarial Robustness Auditing</title></Head>

      {/* Hero */}
      <section className="text-center py-16 md:py-24">
        <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-4">
          <span className="text-[#CC0000]">KAAL</span>
        </h1>
        <p className="text-lg md:text-xl text-gray-400 italic mb-8">
          &ldquo;What cannot be seen, cannot be defended.&rdquo;
        </p>
        <p className="text-gray-300 max-w-xl mx-auto mb-10 leading-relaxed">
          Adversarial robustness auditing for image classification and object
          detection models. Attack your model, score its vulnerability, and
          generate a full report — entirely offline.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link
            href="/audit"
            className="px-6 py-3 bg-[#CC0000] hover:bg-[#aa0000] text-white font-semibold rounded-lg transition-colors focus-visible:ring-2 focus-visible:ring-[#CC0000] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0A0A0A]"
          >
            Start an Audit →
          </Link>
          <Link
            href="/patch"
            className="px-6 py-3 border border-[#333] hover:border-[#555] text-gray-300 hover:text-white font-semibold rounded-lg transition-colors"
          >
            Generate a Patch
          </Link>
          <Link
            href="/compare"
            className="px-6 py-3 border border-[#333] hover:border-[#555] text-gray-300 hover:text-white font-semibold rounded-lg transition-colors"
          >
            Compare Audits
          </Link>
        </div>
      </section>

      {/* Attack modules */}
      <section className="mb-16">
        <h2 className="text-xl font-semibold text-gray-200 mb-6 text-center">Attack Modules</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {ATTACKS.map((a) => (
            <div
              key={a.name}
              className="bg-[#111] border border-[#222] rounded-lg p-5 hover:border-[#333] transition-colors"
            >
              <div className="text-3xl mb-3">{a.icon}</div>
              <div className="font-semibold text-white">{a.name}</div>
              <div className="text-xs text-[#CC0000] mb-2">{a.sub}</div>
              <p className="text-xs text-gray-400 leading-relaxed">{a.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* KVS scale */}
      <section className="mb-16">
        <h2 className="text-xl font-semibold text-gray-200 mb-2 text-center">KVS Score</h2>
        <p className="text-center text-sm text-gray-500 mb-6">
          KAAL Vulnerability Score — 0.0 to 10.0 across five dimensions
        </p>
        <div className="bg-[#111] border border-[#222] rounded-lg overflow-hidden">
          <table className="w-full text-sm" role="table">
            <thead>
              <tr className="border-b border-[#222] text-gray-500 text-xs uppercase tracking-wider">
                <th className="px-4 py-3 text-left" scope="col">Score Range</th>
                <th className="px-4 py-3 text-left" scope="col">Risk Label</th>
              </tr>
            </thead>
            <tbody>
              {KVS_TIERS.map((tier) => (
                <tr key={tier.label} className="border-b border-[#1a1a1a] last:border-0">
                  <td className="px-4 py-3 font-mono text-gray-300">{tier.range}</td>
                  <td className="px-4 py-3">
                    <span
                      className="px-2 py-0.5 rounded-full text-xs font-semibold border"
                      style={{ color: tier.color, borderColor: tier.color, backgroundColor: `${tier.color}18` }}
                    >
                      {tier.label}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
