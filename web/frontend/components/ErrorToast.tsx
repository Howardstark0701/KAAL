import React, { createContext, useCallback, useContext, useState } from 'react';

interface Toast { id: number; message: string; }
interface ToastCtx { showError: (msg: string) => void; }

const ToastContext = createContext<ToastCtx>({ showError: () => {} });

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  let nextId = 0;

  const showError = useCallback((message: string) => {
    const id = ++nextId;
    setToasts((prev) => [...prev, { id, message }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 5000);
  }, []);

  return (
    <ToastContext.Provider value={{ showError }}>
      {children}
      {/* Toast container */}
      <div
        aria-live="assertive"
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            role="alert"
            className="bg-[#1a0000] border border-[#CC0000] text-red-300 text-sm rounded px-4 py-3 flex items-start gap-2 shadow-lg"
          >
            <span className="text-[#CC0000] font-bold mt-0.5 shrink-0">!</span>
            <span className="flex-1">{t.message}</span>
            <button
              aria-label="Dismiss notification"
              onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
              className="text-gray-500 hover:text-gray-300 ml-1 shrink-0"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}
