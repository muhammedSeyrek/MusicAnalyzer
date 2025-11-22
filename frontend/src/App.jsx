import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Music, Disc3, Activity, Sparkles } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import ResultsDisplay from './components/ResultsDisplay';
import WaveformBackground from './components/WaveformBackground';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState('');

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setFileName(file.name);
    setAnalyzing(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 1, 95));
      }, 100);

      const response = await axios.post(`${API_URL}/api/analyze`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        setResult(response.data.analysis);
        setAnalyzing(false);
      }, 500);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
      setAnalyzing(false);
      setProgress(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/mpeg': ['.mp3'],
      'audio/wav': ['.wav'],
      'audio/flac': ['.flac'],
    },
    maxSize: 200 * 1024 * 1024, // 200MB
    multiple: false,
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background */}
      <WaveformBackground />

      {/* Gradient Mesh Overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-purple-900/20 via-transparent to-transparent" />

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-4 py-8 min-h-screen flex flex-col">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <motion.div
              animate={{
                rotate: [0, 360],
              }}
              transition={{
                duration: 20,
                repeat: Infinity,
                ease: 'linear',
              }}
            >
              <Disc3 className="w-12 h-12 text-purple-400" />
            </motion.div>
            <h1 className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400">
              Music Analyzer
            </h1>
          </div>
          <p className="text-xl text-purple-200/80">
            Advanced Pattern Recognition • Western & Eastern Music
          </p>
        </motion.header>

        {/* Upload Area */}
        <AnimatePresence mode="wait">
          {!result && !analyzing && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.4 }}
              className="flex-grow flex items-center justify-center"
            >
              <div
                {...getRootProps()}
                className={`
                  relative group cursor-pointer
                  w-full max-w-2xl p-12
                  rounded-3xl
                  backdrop-blur-xl bg-white/5
                  border-2 border-purple-500/30
                  shadow-2xl shadow-purple-900/50
                  transition-all duration-300
                  hover:bg-white/10 hover:border-purple-400/50 hover:shadow-purple-500/50
                  ${isDragActive ? 'bg-white/10 border-purple-400 scale-105' : ''}
                `}
              >
                <input {...getInputProps()} />

                {/* Glow effect */}
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-purple-600/20 to-pink-600/20 blur-2xl group-hover:blur-3xl transition-all" />

                <div className="relative z-10 text-center">
                  <motion.div
                    animate={{
                      y: [0, -10, 0],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'easeInOut',
                    }}
                    className="mb-6"
                  >
                    <Upload className="w-20 h-20 mx-auto text-purple-400" />
                  </motion.div>

                  <h2 className="text-3xl font-bold text-white mb-4">
                    {isDragActive ? 'Drop it here!' : 'Upload Music File'}
                  </h2>

                  <p className="text-purple-200/80 mb-6">
                    Drag & drop or click to select
                  </p>

                  <div className="flex items-center justify-center gap-4 text-sm text-purple-300/60">
                    <span className="px-3 py-1 rounded-full bg-purple-500/20">MP3</span>
                    <span className="px-3 py-1 rounded-full bg-purple-500/20">WAV</span>
                    <span className="px-3 py-1 rounded-full bg-purple-500/20">FLAC</span>
                    <span className="text-purple-400/60">• Max 200MB</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Analyzing State */}
          {analyzing && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-grow flex items-center justify-center"
            >
              <div className="text-center">
                <motion.div
                  animate={{
                    rotate: 360,
                    scale: [1, 1.1, 1],
                  }}
                  transition={{
                    rotate: { duration: 2, repeat: Infinity, ease: 'linear' },
                    scale: { duration: 1, repeat: Infinity, ease: 'easeInOut' },
                  }}
                  className="mb-8"
                >
                  <Activity className="w-24 h-24 mx-auto text-purple-400" />
                </motion.div>

                <h2 className="text-3xl font-bold text-white mb-4">
                  Analyzing {fileName}
                </h2>

                {/* Progress Bar */}
                <div className="w-96 h-2 bg-white/10 rounded-full overflow-hidden backdrop-blur-sm">
                  <motion.div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>

                <p className="text-purple-200/60 mt-4">{progress}%</p>
              </div>
            </motion.div>
          )}

          {/* Results */}
          {result && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.5 }}
              className="flex-grow"
            >
              <ResultsDisplay result={result} fileName={fileName} />

              {/* Analyze Another Button */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="text-center mt-8"
              >
                <button
                  onClick={() => {
                    setResult(null);
                    setFileName('');
                  }}
                  className="px-8 py-4 rounded-2xl backdrop-blur-xl bg-white/10 border border-purple-500/30 text-white font-semibold hover:bg-white/20 transition-all duration-300 shadow-lg hover:shadow-purple-500/50"
                >
                  <Sparkles className="inline w-5 h-5 mr-2" />
                  Analyze Another Track
                </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-8 left-1/2 transform -translate-x-1/2 px-6 py-4 rounded-2xl backdrop-blur-xl bg-red-500/20 border border-red-500/50 text-white max-w-md"
          >
            <p className="font-semibold">⚠️ {error}</p>
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default App;
