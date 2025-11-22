# Music Analyzer Frontend

Modern React frontend with glassmorphism design and smooth animations.

## Features

- 🎨 Unique glassmorphism design with gradient mesh
- ✨ Smooth Framer Motion animations
- 📤 Drag & drop file upload
- 📊 Beautiful results visualization
- 🌊 Animated waveform background
- 📱 Fully responsive design

## Tech Stack

- React 18
- Vite
- Tailwind CSS
- Framer Motion
- React Dropzone
- Axios
- Lucide React Icons

## Development

```bash
# Install dependencies
npm install

# Start dev server (with API proxy)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## API Integration

The frontend connects to the FastAPI backend at:
- Development: `http://localhost:8080` (via Vite proxy)
- Production: Same origin (served by FastAPI)

## Environment Variables

Create `.env` file for custom API URL:

```env
VITE_API_URL=http://localhost:8080
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ResultsDisplay.jsx    # Results visualization
│   │   └── WaveformBackground.jsx # Animated background
│   ├── App.jsx                    # Main app component
│   ├── App.css                    # Custom styles & animations
│   └── main.jsx                   # Entry point
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

## Design System

**Colors:**
- Primary: Purple gradient (#667eea → #764ba2)
- Accent: Pink gradient (#f093fb → #f5576c)
- Background: Dark slate with gradient overlay

**Effects:**
- Glassmorphism: `backdrop-blur-xl` + `bg-white/5`
- Gradients: Animated gradient backgrounds
- Shadows: Colored glow effects
- Animations: Smooth transitions with Framer Motion

## Performance

- Code splitting with Vite
- Optimized bundle size
- Lazy loading for components
- Efficient re-renders with React best practices
