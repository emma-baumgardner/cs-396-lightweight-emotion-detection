import React, { useState, useRef, useEffect } from 'react';
import { Settings, Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';
// import from our other files

// load models

export default function EmotionalCharades() {
  const [gameState, setGameState] = useState('menu'); // menu, playing, paused
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [timeLeft, setTimeLeft] = useState(60); // can make a game mode, if desired
  const [showSettings, setShowSettings] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false); // need to turn sound on; menu item
  const [difficulty, setDifficulty] = useState('medium'); // default
  const [cameraActive, setCameraActive] = useState(true);
  const videoRef = useRef(null);

  // update to include emotions that are being used
  const [detectedExpressions, setDetectedExpressions] = useState([
    { emotion: 'Happy', confidence: 0 },
    { emotion: 'Fear', confidence: 0 },
    { emotion: 'Disgust', confidence: 0 },
    { emotion: 'Sad', confidence: 0 },
    { emotion: 'Angry', confidence: 0 },
    { emotion: 'Surprised', confidence: 0 },
    { emotion: 'Neutral', confidence: 0 },
  ]);
  // random + seed 
  const [targetEmotion, setTargetEmotion] = useState('Happy');

  // timer 
  useEffect(() => {
    if (gameState !== 'playing') return;
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setGameState('menu');
          return 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [gameState]);

  // camera
  useEffect(() => {
    if (cameraActive && videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
        })
        .catch(err => console.error('Camera error:', err));
    }
  }, [cameraActive]);

  const startGame = () => {
    setCameraActive(true);
    setGameState('playing');
    setTimeLeft(60);
    setScore(0);
    setRound(1);
    setTargetEmotion(getRandomEmotion());
  };

  const pauseGame = () => {
    setGameState(gameState === 'playing' ? 'paused' : 'playing');
  };

  const resetGame = () => {
    setGameState('menu');
    setCameraActive(false);
    setScore(0);
    setRound(1);
    setTimeLeft(60);
  };

  const getRandomEmotion = () => {
    const emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral', 'Fear', 'Disgust'];
    return emotions[Math.floor(Math.random() * emotions.length)];
  };

  // simulate expression detection (replace with real MLP integration)
  useEffect(() => {
    if (gameState !== 'playing') return;
    const interval = setInterval(() => {
      const newExpressions = detectedExpressions.map(exp => ({
        ...exp,
        confidence: Math.random()
      }));
      setDetectedExpressions(newExpressions);

      // Check if target emotion is detected with high confidence
      const targetExp = newExpressions.find(e => e.emotion === targetEmotion);
      if (targetExp && targetExp.confidence > 0.7) {
        setScore(prev => prev + Math.round(targetExp.confidence * 100));
        setRound(prev => prev + 1);
        setTargetEmotion(getRandomEmotion());
      }
    }, 500);
    return () => clearInterval(interval);
  }, [gameState, targetEmotion, detectedExpressions]);

  const getConfidenceColor = (confidence) => {
    if (confidence < 0.33) return 'bg-gray-400';
    if (confidence < 0.66) return 'bg-yellow-400';
    return 'bg-green-500';
  };

  const emojiDictionary = (emotion) => {
    const emojis = {
      Happy: '😊',
      Sad: '😢',
      Angry: '😠',
      Surprised: '😮',
      Neutral: '😐',
      Fear: '😬', 
      Disgust: '🤮'
    };
    return emojis[emotion] || '😐';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold">Emotional Charades</h1>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-3 hover:bg-white/10 rounded-lg transition"
          >
            <Settings size={24} />
          </button>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-6 mb-6 border border-white/20">
            <h2 className="text-xl font-bold mb-4">Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-2">Difficulty</label>
                <select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  className="bg-white/10 border border-white/20 rounded px-3 py-2 w-full"
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
              <button
                onClick={() => setSoundEnabled(!soundEnabled)}
                className="flex items-center gap-2 p-2 hover:bg-white/10 rounded transition"
              >
                {soundEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
                {soundEnabled ? 'Sound On' : 'Sound Off'}
              </button>
            </div>
          </div>
        )}

        {gameState === 'menu' ? (
          // Menu Screen
          <div className="text-center space-y-8">
            <div className="text-6xl">🎭</div>
            <h2 className="text-3xl font-bold">Ready to Show Your Emotions?</h2>
            <p className="text-xl text-gray-300">Match the target expression as accurately as possible!</p>
            <button
              onClick={startGame}
              className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700 px-8 py-4 rounded-lg text-lg font-bold transition transform hover:scale-105"
            >
              Start Game
            </button>
          </div>
        ) : (
          // Game Screen
          <div className="space-y-6">
            {/* Stats Bar */}
            <div className="flex justify-between items-center bg-white/10 backdrop-blur-md rounded-lg p-4 border border-white/20">
              <div className="text-center">
                <p className="text-sm text-gray-300">Score</p>
                <p className="text-3xl font-bold">{score}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-300">Round</p>
                <p className="text-3xl font-bold">{round}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-300">Time Left</p>
                <p className="text-3xl font-bold text-yellow-400">{timeLeft}s</p>
              </div>
            </div>

            {/* Target Emotion */}
            <div className="text-center bg-gradient-to-r from-pink-500/20 to-purple-500/20 backdrop-blur-md rounded-lg p-6 border border-white/20">
              <p className="text-sm text-gray-300 mb-2">Target Emotion</p>
              <p className="text-5xl mb-2">{emojiDictionary(targetEmotion)}</p>
              <p className="text-2xl font-bold">{targetEmotion}</p>
            </div>

            {/* Main Content Grid */}
            <div className="grid md:grid-cols-3 gap-6">
              {/* Camera Feed */}
              <div className="md:col-span-2">
                <div className="bg-black rounded-lg overflow-hidden aspect-video border-2 border-white/20">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              {/* Expression Confidence Scores */}
              <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 border border-white/20 space-y-3">
                <h3 className="font-bold text-lg mb-4">Detected Emotions</h3>
                {detectedExpressions.map((exp) => (
                  <div key={exp.emotion}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm">{exp.emotion}</span>
                      <span className="text-xs text-gray-300">{Math.round(exp.confidence * 100)}%</span>
                    </div>
                    <div className="bg-white/10 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full transition-all ${getConfidenceColor(exp.confidence)}`}
                        style={{ width: `${exp.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="flex justify-center gap-4">
              <button
                onClick={pauseGame}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-bold transition"
              >
                {gameState === 'playing' ? <Pause size={20} /> : <Play size={20} />}
                {gameState === 'playing' ? 'Pause' : 'Resume'}
              </button>
              <button
                onClick={resetGame}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-bold transition"
              >
                <RotateCcw size={20} />
                Exit
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}