/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, RefreshCw, Palette, AlertCircle } from 'lucide-react';

const PERSONAS = [
  { name: 'None', value: 'transparent' },
  { name: 'Chief Engineer', value: '#F59E0B' },
  { name: 'Assistant to the Manager', value: '#94A3B8' },
  { name: 'Intern Energy', value: '#4ADE80' },
  { name: 'Supervisor', value: '#EF4444' },
  { name: 'Expert Mode', value: '#06B6D4' },
  { name: 'Field Guy', value: '#D97706' },
  { name: 'Control Room Vibes', value: '#8B5CF6' },
  { name: 'Looks Busy Mode', value: '#F472B6' },
  { name: 'Professional Thinker', value: '#6366F1' },
  { name: 'Senior Button Clicker', value: '#10B981' },
];

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageSegmenter, setImageSegmenter] = useState<ImageSegmenter | null>(null);
  const [selectedColor, setSelectedColor] = useState(PERSONAS[1].value);
  const [intensity, setIntensity] = useState(0.8);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isWebcamStarted, setIsWebcamStarted] = useState(false);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  
  const requestRef = useRef<number>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const prevMaskRef = useRef<Float32Array | null>(null);
  const lastManualSelectionRef = useRef<number>(0);

  const currentPersonaName = PERSONAS.find(p => p.value === selectedColor)?.name || 'Standard';

  // Automatic Persona Assignment on Detection
  useEffect(() => {
    if (isFaceDetected) {
      // Trigger scan immediately on reappearance
      if (!isScanning) {
        triggerNeuralScan();
      }
    } else {
      // Reset to None if face is lost
      setSelectedColor(PERSONAS[0].value);
      setIsScanning(false);
    }
  }, [isFaceDetected]);

  const triggerNeuralScan = () => {
    setIsScanning(true);
    let iterations = 0;
    const maxIterations = 20; // Slightly longer for better visibility
    
    const scanInterval = setInterval(() => {
      const randomIndex = Math.floor(Math.random() * (PERSONAS.length - 1)) + 1;
      setSelectedColor(PERSONAS[randomIndex].value);
      iterations++;
      
      if (iterations >= maxIterations) {
        clearInterval(scanInterval);
        setIsScanning(false);
      }
    }, 80); // Slightly slower for a more "analytical" feel
  };

  const handleManualSelection = (color: string) => {
    lastManualSelectionRef.current = Date.now();
    setSelectedColor(color);
    setIsScanning(false);
  };

  useEffect(() => {
    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
        );
        
        // Load Multi-class Selfie Segmenter
        // Categories: 0-bg, 1-hair, 2-body, 3-face, 4-clothes, 5-others
        const segmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputCategoryMask: false,
          outputConfidenceMasks: true
        });
        
        setImageSegmenter(segmenter);
        setIsLoading(false);
      } catch (err) {
        console.error("Failed to initialize MediaPipe:", err);
        setError("Failed to load AI models. Please check your connection.");
        setIsLoading(false);
      }
    }

    initMediaPipe();

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      imageSegmenter?.close();
    };
  }, []);

  const startWebcam = async () => {
    if (!videoRef.current) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, facingMode: 'user' } 
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play();
        setIsWebcamStarted(true);
      };
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError("Webcam access denied. Please enable camera permissions.");
    }
  };

  useEffect(() => {
    if (isWebcamStarted && imageSegmenter) {
      renderLoop();
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isWebcamStarted, imageSegmenter, selectedColor, intensity]);

  const renderLoop = () => {
    if (!videoRef.current || !canvasRef.current || !imageSegmenter) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (!ctx || video.paused || video.ended) {
      requestRef.current = requestAnimationFrame(renderLoop);
      return;
    }

    // Ensure canvas matches video dimensions
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      if (video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        if (!maskCanvasRef.current) {
          maskCanvasRef.current = document.createElement('canvas');
        }
        maskCanvasRef.current.width = video.videoWidth;
        maskCanvasRef.current.height = video.videoHeight;
        prevMaskRef.current = null;
      }
    }

    if (canvas.width === 0) {
      requestRef.current = requestAnimationFrame(renderLoop);
      return;
    }

    const startTimeMs = performance.now();
    
    // 1. Run Multi-class Segmentation
    imageSegmenter.segmentForVideo(video, startTimeMs, (result) => {
      const confidenceMasks = result.confidenceMasks;
      if (!confidenceMasks || confidenceMasks.length < 4) return;

      // Index 1 is hair, Index 3 is face in the multiclass model
      const hairMask = confidenceMasks[1];
      const faceMask = confidenceMasks[3];

      // Check for face presence (simple average of a central block or sum)
      const faceData = faceMask.getAsFloat32Array();
      let faceConfidenceSum = 0;
      // Sampling every 100th pixel for performance to check existence
      for(let i = 0; i < faceData.length; i += 100) {
        faceConfidenceSum += faceData[i];
      }
      const detected = faceConfidenceSum > 20; 
      if (detected !== isFaceDetected) {
        setIsFaceDetected(detected);
      }

      // 2. Clear main canvas and draw video
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (selectedColor !== 'transparent') {
        const maskCanvas = maskCanvasRef.current!;
        const maskCtx = maskCanvas.getContext('2d')!;
        
        const hairData = hairMask.getAsFloat32Array();
        
        // Adjusted smoothing factor to 0.4 for a balance between responsiveness and stability
        if (!prevMaskRef.current || prevMaskRef.current.length !== hairData.length) {
          prevMaskRef.current = new Float32Array(hairData);
        } else {
          const smoothingFactor = 0.4;
          for (let i = 0; i < hairData.length; i++) {
            prevMaskRef.current[i] = (hairData[i] * smoothingFactor) + (prevMaskRef.current[i] * (1 - smoothingFactor));
          }
        }

        const smoothedMask = prevMaskRef.current;
        const imageData = maskCtx.createImageData(canvas.width, canvas.height);
        
        for (let i = 0; i < smoothedMask.length; i++) {
          const confidence = smoothedMask[i];
          const pixelIndex = i * 4;
          
          // Soft Alpha Ramping: Start at 30% confidence to catch fine flyaways
          // We use a wider ramp (0.3 to 0.7) for a smoother transition
          let alpha = 0;
          if (confidence > 0.3) {
            alpha = Math.min(255, ((confidence - 0.3) / 0.4) * 255 * intensity);
          }
          
          imageData.data[pixelIndex] = 255;
          imageData.data[pixelIndex + 1] = 255;
          imageData.data[pixelIndex + 2] = 255;
          imageData.data[pixelIndex + 3] = alpha;
        }
        
        maskCtx.putImageData(imageData, 0, 0);

        // 4. Color the mask
        maskCtx.globalCompositeOperation = 'source-in';
        maskCtx.fillStyle = selectedColor;
        maskCtx.fillRect(0, 0, canvas.width, canvas.height);
        maskCtx.globalCompositeOperation = 'source-over';

        // 5. Blend colored mask onto main canvas
        // We use a slightly stronger blur (2px) to "dilate" the mask and catch flyaways
        ctx.save();
        ctx.globalCompositeOperation = 'soft-light';
        ctx.filter = 'blur(2px)'; 
        ctx.drawImage(maskCanvas, 0, 0);
        ctx.restore();
      }
      ctx.restore();
    });

    requestRef.current = requestAnimationFrame(renderLoop);
  };

  return (
    <div className="min-h-screen relative bg-[#090510] text-white font-sans flex flex-col items-center justify-center p-4 overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 z-0 pointer-events-none opacity-25" 
           style={{ backgroundImage: 'radial-gradient(circle, #6366f1 1px, transparent 1px)', backgroundSize: '50px 50px' }} />

      {/* Dynamic Background Atmosphere */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[70%] h-[70%] rounded-full bg-indigo-600/25 blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[70%] h-[70%] rounded-full bg-purple-600/25 blur-[120px] animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-[20%] right-[10%] w-[50%] h-[50%] rounded-full bg-pink-600/15 blur-[100px]" />
      </div>

      {/* Top Bar */}
      <div className="z-10 w-full max-w-4xl mb-8 flex flex-col md:flex-row items-center md:items-end justify-between gap-6 md:gap-0 px-10 py-8 bg-white/10 backdrop-blur-2xl rounded-[2.5rem] border border-white/20 shadow-2xl">
        <div className="flex flex-col items-center md:items-start text-center md:text-left">
          <h1 className="text-5xl font-black tracking-tighter uppercase italic bg-gradient-to-r from-white to-white/40 bg-clip-text text-transparent flex items-center gap-4">
            <Palette className="w-10 h-10 text-purple-400 drop-shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
            Grid Hair
          </h1>
        </div>
        
        <div className="flex gap-4 mb-1">
          <div className="px-5 py-2.5 bg-black/40 backdrop-blur-md border border-white/20 rounded-full font-mono text-[10px] uppercase tracking-wider flex items-center gap-3 shadow-lg">
            <div className={`w-2.5 h-2.5 rounded-full ${isWebcamStarted ? 'bg-purple-400 animate-pulse shadow-[0_0_12px_rgba(192,132,252,1)]' : 'bg-red-500'}`} />
            <span className={isWebcamStarted ? 'text-purple-300 font-bold' : 'text-red-400'}>{isWebcamStarted ? 'Link Active' : 'Offline'}</span>
          </div>
          <div className="px-5 py-2.5 bg-black/40 backdrop-blur-md border border-white/20 rounded-full font-mono text-[10px] uppercase tracking-wider text-white/50 shadow-lg">
            RT-SCAN
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      <div className="z-10 relative w-full max-w-4xl aspect-video bg-black/60 rounded-[3rem] border border-white/20 shadow-[0_45px_100px_-20px_rgba(0,0,0,1)] overflow-hidden group">
        {/* Scanning Lines */}
        {isWebcamStarted && isFaceDetected && (
          <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500/50 to-transparent animate-[scan_3s_linear_infinite]" />
            <style>{`
              @keyframes scan {
                0% { transform: translateY(-100%); }
                100% { transform: translateY(800px); }
              }
            `}</style>
          </div>
        )}
        
        <video 
          ref={videoRef} 
          className="hidden" 
          autoPlay 
          playsInline 
          muted 
        />
        
        <canvas 
          ref={canvasRef} 
          className={`w-full h-full object-cover transition-all duration-1000 ${isScanning ? 'scale-[1.02] brightness-125' : ''}`}
          style={{ transform: 'scaleX(-1)', opacity: isWebcamStarted ? 1 : 0 }}
        />

        {/* Dynamic Persona Badge */}
        {isWebcamStarted && (
          <div className="absolute top-10 left-10 z-20 flex flex-col gap-3 animate-in fade-in slide-in-from-left-6 duration-700">
            {isFaceDetected ? (
              <div 
                className={`px-8 py-4 rounded-[1.5rem] font-black italic uppercase tracking-[0.3em] text-lg flex items-center gap-4 backdrop-blur-3xl border-l-[6px] shadow-2xl transition-all duration-500 ${isScanning ? 'scale-105 animate-pulse' : ''}`}
                style={{ 
                  backgroundColor: `${selectedColor}44`, 
                  borderColor: selectedColor,
                  color: '#ffffff',
                  textShadow: selectedColor !== 'transparent' ? `0 0 15px ${selectedColor}` : 'none'
                }}
              >
                <div className={`w-3 h-3 rounded-full ${isScanning ? 'animate-spin' : 'animate-ping'}`} style={{ backgroundColor: selectedColor }} />
                {isScanning ? 'Neural Scan in Progress...' : `Active Role: ${currentPersonaName}`}
              </div>
            ) : (
              <div className="px-8 py-4 rounded-[1.5rem] border-white/20 bg-black/40 backdrop-blur-2xl text-white/40 font-black italic uppercase tracking-[0.3em] text-lg flex items-center gap-4 border-2 shadow-2xl">
                <div className="w-3 h-3 rounded-full bg-red-500/50 animate-pulse" />
                Active Role: None
              </div>
            )}
            <div className={`px-5 py-2 backdrop-blur-xl rounded-full border w-fit text-[11px] font-mono uppercase tracking-[0.4em] shadow-lg font-bold transition-all duration-500 ${isFaceDetected ? 'bg-black/60 border-white/20 text-purple-300/80' : 'bg-white/5 border-white/5 text-white/10'}`}>
              {isFaceDetected ? 'Biometric ID Confirmed' : 'Neural Grid: Standby'}
            </div>
          </div>
        )}

        {/* System State Overlay */}
        {(!isWebcamStarted && !error) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md z-20 gap-12">
            {!isLoading && (
              <button 
                onClick={startWebcam}
                className="group relative px-12 py-5 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-full font-black uppercase italic tracking-tighter text-xl hover:scale-105 transition-all duration-300 flex items-center gap-4 shadow-[0_0_40px_rgba(147,51,234,0.4)] active:scale-95"
              >
                <Camera className="w-7 h-7" />
                Launch Vision
              </button>
            )}
            
            <div className="flex flex-col items-center">
              <RefreshCw className={`w-14 h-14 text-purple-400 mb-6 ${isLoading ? 'animate-spin' : 'opacity-20'}`} />
              <p className="font-mono text-[11px] uppercase tracking-[0.4em] text-purple-400/80">
                {isLoading ? 'Calibrating Neural Models...' : 'Ready for Feed Initialization'}
              </p>
            </div>
          </div>
        )}

        {/* Error Overlay */}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-950/80 backdrop-blur-xl z-30 p-8 text-center">
            <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mb-6">
              <AlertCircle className="w-12 h-12 text-red-500" />
            </div>
            <h2 className="text-2xl font-black mb-2 uppercase tracking-tight italic">Engine Malfunction</h2>
            <p className="text-red-200/60 max-w-md text-sm mb-8 font-medium leading-relaxed">{error}</p>
            <button 
              onClick={() => window.location.reload()}
              className="px-10 py-3 bg-red-500 hover:bg-red-400 text-white rounded-full font-black text-xs uppercase tracking-[0.2em] transition-all shadow-[0_0_30px_rgba(239,68,68,0.4)]"
            >
              Restart System
            </button>
          </div>
        )}

      </div>

      {/* Control Panel */}
      <div className="z-10 mt-10 w-full max-w-4xl bg-white/5 backdrop-blur-2xl border border-white/10 rounded-[2.5rem] p-8 shadow-2xl relative">
        <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-6 py-1.5 bg-gradient-to-r from-purple-500 to-indigo-500 text-white font-mono text-[10px] font-black uppercase tracking-[0.3em] rounded-full shadow-lg">
          Chroma Matrix: Staff Protocol
        </div>
        
        <div className="flex flex-wrap justify-center items-center gap-x-10 gap-y-12">
          {PERSONAS.map((color) => (
            <button
              key={color.name}
              onClick={() => handleManualSelection(color.value)}
              className={`group relative flex flex-col items-center gap-3 transition-all duration-300 ${
                selectedColor === color.value ? 'scale-110' : 'opacity-40 hover:opacity-100 hover:translate-y-[-4px]'
              }`}
            >
              <div 
                className={`w-14 h-14 rounded-xl border-2 flex items-center justify-center transition-all duration-300 ${
                  selectedColor === color.value 
                    ? 'border-white shadow-[0_0_25px_rgba(255,255,255,0.4)]' 
                    : 'border-white/5 hover:border-white/20'
                }`}
                style={{ 
                  backgroundColor: color.value === 'transparent' ? 'rgba(255,255,255,0.05)' : color.value,
                  boxShadow: selectedColor === color.value && color.value !== 'transparent' ? `0 0 30px ${color.value}66` : 'none',
                  borderRadius: '16px'
                }}
              >
                {color.value === 'transparent' && (
                  <div className="w-8 h-0.5 bg-red-500/40 rotate-45" />
                )}
              </div>
              <div className="flex flex-col items-center gap-1">
                <span className={`text-[10px] text-center max-w-[80px] font-black uppercase tracking-tighter leading-none transition-colors duration-300 ${
                  selectedColor === color.value ? 'text-white' : 'text-white/20 group-hover:text-white/60'
                }`}>
                  {color.name}
                </span>
              </div>
              {selectedColor === color.value && (
                <div className="absolute -bottom-2 w-1.5 h-1.5 bg-white rounded-full animate-bounce" />
              )}
            </button>
          ))}
        </div>

        {/* Intensity Controller */}
        <div className="mt-14 w-full max-w-md mx-auto flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
          <div className="flex justify-between items-end px-2">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/40">Density Level</span>
              <span className="text-[9px] font-mono text-purple-400/60 uppercase">Chroma saturation intensity</span>
            </div>
            <span className="text-xl font-mono text-white tracking-widest leading-none">
              {(intensity * 100).toFixed(0)}<span className="text-[10px] text-white/30 ml-1">%</span>
            </span>
          </div>
          <div className="relative group/slider px-2">
            <div className="absolute inset-x-0 h-1 bg-white/5 rounded-full" />
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.01" 
              value={intensity} 
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              className="relative w-full h-1 bg-transparent appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400 transition-all z-10"
              style={{
                background: `linear-gradient(to right, #a855f7 0%, #a855f7 ${intensity * 100}%, transparent ${intensity * 100}%)`
              }}
            />
            <div className="absolute -inset-2 bg-purple-500/10 blur-xl rounded-full opacity-0 group-hover/slider:opacity-100 transition-opacity" />
          </div>
          <div className="flex justify-between px-2 text-[8px] font-mono uppercase tracking-widest text-white/10">
            <span>Minimum</span>
            <span>Neural Peak</span>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="z-10 mt-12 flex flex-wrap justify-center gap-x-12 gap-y-4 text-[10px] font-mono uppercase tracking-[0.3em] text-white/20">
        <div className="flex items-center gap-3">
          <span className="text-white/10 decoration-white/20 underline underline-offset-4 font-bold">Model</span>
          <span className="text-purple-400/40">Multi-Class Selfie v2</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-white/10 decoration-white/20 underline underline-offset-4 font-bold">Process</span>
          <span className="text-indigo-400/40">Canvas Pipeline 2D</span>
        </div>
        <div className="flex items-center gap-3 text-white/40 font-black italic">
          v1.4.0-STABLE
        </div>
      </div>
    </div>
  );
}
