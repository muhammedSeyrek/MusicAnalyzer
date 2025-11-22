import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const WaveformBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let time = 0;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Waveform parameters
    const waves = [
      { amplitude: 50, frequency: 0.002, speed: 0.5, color: 'rgba(147, 51, 234, 0.1)' },
      { amplitude: 40, frequency: 0.003, speed: 0.7, color: 'rgba(168, 85, 247, 0.08)' },
      { amplitude: 60, frequency: 0.0015, speed: 0.3, color: 'rgba(192, 132, 252, 0.06)' },
    ];

    const drawWave = (wave, offset) => {
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);

      for (let x = 0; x < canvas.width; x++) {
        const y =
          canvas.height / 2 +
          Math.sin((x + offset) * wave.frequency) * wave.amplitude +
          Math.sin((x + offset * 0.5) * wave.frequency * 2) * (wave.amplitude * 0.3);

        ctx.lineTo(x, y);
      }

      ctx.strokeStyle = wave.color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Fill below the wave
      ctx.lineTo(canvas.width, canvas.height);
      ctx.lineTo(0, canvas.height);
      ctx.closePath();
      ctx.fillStyle = wave.color;
      ctx.fill();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw multiple waves
      waves.forEach((wave) => {
        drawWave(wave, time * wave.speed * 10);
      });

      time += 1;
      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <>
      {/* Animated canvas */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 0 }}
      />

      {/* Floating orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
        <motion.div
          animate={{
            x: [0, 100, 0],
            y: [0, -100, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
        />

        <motion.div
          animate={{
            x: [0, -100, 0],
            y: [0, 100, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl"
        />

        <motion.div
          animate={{
            x: [0, 50, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="absolute top-1/2 right-1/3 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"
        />
      </div>

      {/* Grid pattern overlay */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          zIndex: 0,
          backgroundImage: `
            linear-gradient(rgba(147, 51, 234, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(147, 51, 234, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />
    </>
  );
};

export default WaveformBackground;
