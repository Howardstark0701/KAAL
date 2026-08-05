import React, { useId, useRef, useState } from 'react';

interface UploadZoneProps {
  id?: string;
  label: string;
  accept: string;
  multiple?: boolean;
  onFiles: (files: File[]) => void;
  status?: 'idle' | 'uploading' | 'success' | 'error';
  statusMessage?: string;
  disabled?: boolean;
}

export default function UploadZone({
  label,
  accept,
  multiple = false,
  onFiles,
  status = 'idle',
  statusMessage,
  disabled = false,
}: UploadZoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const borderColor =
    status === 'success' ? 'border-green-500'  :
    status === 'error'   ? 'border-red-500'     :
    dragging             ? 'border-[#CC0000]'   :
                           'border-[#333]';

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    onFiles(Array.from(files));
  };

  return (
    <div>
      <label
        htmlFor={inputId}
        className={`
          flex flex-col items-center justify-center
          min-h-[140px] w-full rounded-lg border-2 border-dashed
          cursor-pointer transition-all duration-150 select-none
          ${borderColor}
          ${disabled ? 'opacity-40 cursor-not-allowed' : 'hover:border-[#CC0000] hover:bg-white/[0.02]'}
          bg-[#111] p-4 text-center
        `}
        aria-label={label}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) handleFiles(e.dataTransfer.files);
        }}
      >
        <span className="text-2xl mb-2">📂</span>
        <span className="text-sm font-medium text-gray-300">{label}</span>
        <span className="text-xs text-gray-500 mt-1">Drag & drop or click to browse</span>
        {statusMessage && (
          <span className={`text-xs mt-2 ${status === 'error' ? 'text-red-400' : 'text-green-400'}`}>
            {statusMessage}
          </span>
        )}
      </label>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        disabled={disabled}
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
        aria-label={label}
      />
    </div>
  );
}
