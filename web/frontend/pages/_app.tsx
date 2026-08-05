import type { AppProps } from 'next/app';
import Layout from '../components/Layout';
import { ToastProvider } from '../components/ErrorToast';
import '../styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ToastProvider>
      <Layout>
        <Component {...pageProps} />
      </Layout>
    </ToastProvider>
  );
}
