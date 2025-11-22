import { motion } from 'framer-motion';
import { Music, TrendingUp, Clock, Disc, Sparkles, BarChart3 } from 'lucide-react';

const ResultsDisplay = ({ result, fileName }) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.1,
        duration: 0.5,
        ease: 'easeOut',
      },
    }),
  };

  const StatCard = ({ icon: Icon, label, value, subtitle, gradient, index }) => (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      className="relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-br opacity-50 blur-xl group-hover:opacity-70 transition-opacity rounded-2xl"
           style={{ background: gradient }} />

      <div className="relative backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-6
                      shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02]">
        <div className="flex items-center justify-between mb-4">
          <Icon className="w-8 h-8 text-white/90" />
          <span className="text-xs text-white/60 uppercase tracking-wider">{label}</span>
        </div>

        <div className="space-y-1">
          <div className="text-3xl font-bold text-white">{value}</div>
          {subtitle && (
            <div className="text-sm text-white/70">{subtitle}</div>
          )}
        </div>
      </div>
    </motion.div>
  );

  const GenreCard = ({ genre, confidence, allGenres }) => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="relative overflow-hidden rounded-3xl mb-8"
    >
      {/* Animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-500 to-purple-600
                      animate-gradient-x" />

      {/* Content */}
      <div className="relative backdrop-blur-sm bg-black/20 p-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <Sparkles className="w-8 h-8 text-white" />
          <h3 className="text-4xl font-bold text-white">{genre}</h3>
          <Sparkles className="w-8 h-8 text-white" />
        </div>

        <div className="flex items-center justify-center gap-6 text-white/90">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            <span className="text-lg font-semibold">
              {(confidence * 100).toFixed(0)}% Confidence
            </span>
          </div>

          {allGenres && allGenres.length > 1 && (
            <div className="flex gap-2">
              {allGenres.slice(0, 3).map((g, idx) => (
                <span key={idx} className="px-3 py-1 rounded-full bg-white/20 text-sm">
                  {g}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );

  const InstrumentBadge = ({ instrument, index }) => (
    <motion.span
      initial={{ opacity: 0, scale: 0 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.5 + index * 0.05 }}
      className="px-4 py-2 rounded-full backdrop-blur-md bg-purple-500/20 border border-purple-400/30
                 text-purple-200 text-sm font-medium hover:bg-purple-500/30 transition-colors"
    >
      {instrument}
    </motion.span>
  );

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h2 className="text-4xl font-bold text-white mb-2">Analysis Complete</h2>
        <p className="text-purple-200/80 text-lg">{fileName}</p>
      </motion.div>

      {/* Genre Card (if available) */}
      {result.genre && result.genre.primary_genre !== 'Unknown' && (
        <GenreCard
          genre={result.genre.primary_genre}
          confidence={result.genre.confidence}
          allGenres={result.genre.all_genres}
        />
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={Music}
          label="Music System"
          value={result.tonality.is_western ? 'Western' : 'Eastern'}
          subtitle={`${(result.tonality.confidence * 100).toFixed(0)}% confident`}
          gradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          index={0}
        />

        <StatCard
          icon={Disc}
          label={result.tonality.is_western ? 'Scale/Key' : 'Makam'}
          value={result.tonality.is_western ? result.tonality.western_tonality : result.tonality.eastern_makam}
          subtitle={result.tonality.is_western ? 'Western Tonality' : 'Eastern Makam'}
          gradient="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
          index={1}
        />

        <StatCard
          icon={Clock}
          label="Tempo"
          value={`${result.rhythm.tempo.toFixed(0)} BPM`}
          subtitle={`${result.rhythm.meter} time`}
          gradient="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
          index={2}
        />

        <StatCard
          icon={TrendingUp}
          label="Complexity"
          value={result.rhythm.complexity}
          subtitle={`${(result.rhythm.regularity * 100).toFixed(0)}% regular`}
          gradient="linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
          index={3}
        />
      </div>

      {/* Detailed Info Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Instruments */}
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-xl">
          <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Music className="w-6 h-6" />
            Detected Instruments
          </h3>

          {result.timbre.instruments && result.timbre.instruments.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {result.timbre.instruments.map((instrument, idx) => (
                <InstrumentBadge key={idx} instrument={instrument} index={idx} />
              ))}
            </div>
          ) : (
            <p className="text-purple-200/60">No specific instruments detected</p>
          )}
        </div>

        {/* Audio Characteristics */}
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-xl">
          <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-6 h-6" />
            Audio Characteristics
          </h3>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-purple-200/80">Brightness</span>
                <span className="text-white font-semibold">
                  {(result.timbre.brightness * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.timbre.brightness * 100}%` }}
                  transition={{ delay: 0.7, duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-purple-200/80">Harmonic Content</span>
                <span className="text-white font-semibold">
                  {(result.timbre.harmonic_ratio * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.timbre.harmonic_ratio * 100}%` }}
                  transition={{ delay: 0.8, duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-purple-200/80">Percussive Content</span>
                <span className="text-white font-semibold">
                  {(result.timbre.percussive_ratio * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.timbre.percussive_ratio * 100}%` }}
                  transition={{ delay: 0.9, duration: 0.8 }}
                  className="h-full bg-gradient-to-r from-red-400 to-pink-500 rounded-full"
                />
              </div>
            </div>

            {result.tonality.microtonal_ratio !== undefined && (
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-purple-200/80">Microtonal Content</span>
                  <span className="text-white font-semibold">
                    {(result.tonality.microtonal_ratio * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${result.tonality.microtonal_ratio * 100}%` }}
                    transition={{ delay: 1.0, duration: 0.8 }}
                    className="h-full bg-gradient-to-r from-green-400 to-teal-500 rounded-full"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Technical Stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-xl"
      >
        <h3 className="text-xl font-bold text-white mb-4">Technical Details</h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-purple-300">
              {result.duration ? result.duration.toFixed(1) : 'N/A'}s
            </div>
            <div className="text-sm text-purple-200/60">Duration</div>
          </div>

          <div>
            <div className="text-2xl font-bold text-purple-300">
              {result.sample_rate ? (result.sample_rate / 1000).toFixed(1) : 'N/A'}kHz
            </div>
            <div className="text-sm text-purple-200/60">Sample Rate</div>
          </div>

          <div>
            <div className="text-2xl font-bold text-purple-300">
              {result.stats?.total_frequencies || 'N/A'}
            </div>
            <div className="text-sm text-purple-200/60">Frequencies</div>
          </div>

          <div>
            <div className="text-2xl font-bold text-purple-300">
              {result.stats?.microtonal_intervals || 'N/A'}
            </div>
            <div className="text-sm text-purple-200/60">Microtonal</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ResultsDisplay;
