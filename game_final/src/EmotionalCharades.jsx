// use react + lucide-react + onnx runtime web lib
import React, { useState, useRef, useEffect } from 'react';
import { Settings, Play, Pause, RotateCcw, Trophy, Home } from 'lucide-react';
import * as ort from 'onnxruntime-web';

export default function EmotionalCharades() {

  // game flow 
  const [gameState, setGameState] = useState('menu'); // menu, playing, paused, stats
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [timeLeft, setTimeLeft] = useState(60);

  // stats tracking
  const [roundStats, setRoundStats] = useState({
    emotionsMatched: 0,
    totalAttempts: 0,
    averageConfidence: 0,
    bestEmotion: '',
    timeUsed: 0
  });

  // leaderboard
  const [leaderboard, setLeaderboard] = useState([]);
  const [showNameInput, setShowNameInput] = useState(false);
  const [playerName, setPlayerName] = useState('');

  // ui settings
  const [showSettings, setShowSettings] = useState(false);
  const [difficulty, setDifficulty] = useState('medium');

  // camera 
  const [cameraActive, setCameraActive] = useState(true);

  // model and flags
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState(null);

  // refs (to prevent re-renders)
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  
  const confidenceSum = useRef(0);
  const matchCount = useRef(0);
  const detectionWindow = useRef([]);
  const WINDOW_SIZE = 50;
  const MATCH_THRESHOLD = 0.6;
  const CONSISTENCY_THRESHOLD = 0.7; // 70% of frames must match

  // intialize detection array
  const [detectedExpressions, setDetectedExpressions] = useState([
    { emotion: 'Happy', confidence: 0 },
    { emotion: 'Fear', confidence: 0 },
    { emotion: 'Disgust', confidence: 0 },
    { emotion: 'Sad', confidence: 0 },
    { emotion: 'Angry', confidence: 0 },
    { emotion: 'Surprised', confidence: 0 },
    { emotion: 'Neutral', confidence: 0 },
  ]);

  // windowing progress
  const [windowProgress, setWindowProgress] = useState(0);

  // emotion labels (default emotion: happy)
  const emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'];
  const [targetEmotion, setTargetEmotion] = useState('Happy');
  const bestEmotionRef = useRef({ emotion: '', confidence: 0 });

  // feature names 
  const EYE_BLENDSHAPES = [
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight'
  ];
  
  const MOUTH_BLENDSHAPES = [
    'jawOpen',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthPucker', 'mouthFunnel',
    'mouthStretchLeft', 'mouthStretchRight'
  ];

  const BROW_BLENDSHAPES = [
    'browDownLeft', 'browDownRight',
    'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight'
  ];

  const KEY_BLENDSHAPES = [...EYE_BLENDSHAPES, ...MOUTH_BLENDSHAPES, ...BROW_BLENDSHAPES];

  const SCALER_PARAMS = {
    mean: [-0.223004, 0.004080, -0.062604, -0.376583, 1.385455, 0.204377, 0.198454, 0.485745, 0.368301, 0.010791, 0.014236, 0.132943, 0.166108, 0.164922, 0.008094, 0.009898, 0.183989, 0.018988, 0.016543, 0.032666, 0.024699, 0.050553, 0.292111, 0.426540, 0.330073],
    std: [0.127195, 0.174133, 4.255152, 7.613988, 6.459692, 0.173503, 0.196143, 0.169232, 0.184486, 0.018017, 0.030435, 0.194819, 0.286062, 0.290739, 0.028944, 0.036069, 0.291776, 0.035378, 0.041987, 0.072426, 0.069380, 0.120217, 0.289932, 0.279136, 0.284749]
  };

  const standardizeFeatures = (features) => {
    return features.map((val, idx) => 
      (val - SCALER_PARAMS.mean[idx]) / SCALER_PARAMS.std[idx]
    );
  };

  const estimateOpenFaceGaze = (faceLandmarks) => {
    const LEFT_IRIS_CENTER = 468;
    const RIGHT_IRIS_CENTER = 473;
    const LEFT_EYE_INNER = 133;
    const LEFT_EYE_OUTER = 33;
    const RIGHT_EYE_INNER = 362;
    const RIGHT_EYE_OUTER = 263;
    
    const leftEyeCenter = {
      x: (faceLandmarks[LEFT_EYE_INNER].x + faceLandmarks[LEFT_EYE_OUTER].x) / 2,
      y: (faceLandmarks[LEFT_EYE_INNER].y + faceLandmarks[LEFT_EYE_OUTER].y) / 2,
      z: (faceLandmarks[LEFT_EYE_INNER].z + faceLandmarks[LEFT_EYE_OUTER].z) / 2
    };
    
    const rightEyeCenter = {
      x: (faceLandmarks[RIGHT_EYE_INNER].x + faceLandmarks[RIGHT_EYE_OUTER].x) / 2,
      y: (faceLandmarks[RIGHT_EYE_INNER].y + faceLandmarks[RIGHT_EYE_OUTER].y) / 2,
      z: (faceLandmarks[RIGHT_EYE_INNER].z + faceLandmarks[RIGHT_EYE_OUTER].z) / 2
    };
    
    const leftIris = faceLandmarks[LEFT_IRIS_CENTER];
    const rightIris = faceLandmarks[RIGHT_IRIS_CENTER];
    
    const leftGazeVec = {
      x: leftIris.x - leftEyeCenter.x,
      y: leftIris.y - leftEyeCenter.y,
      z: leftIris.z - leftEyeCenter.z
    };
    
    const rightGazeVec = {
      x: rightIris.x - rightEyeCenter.x,
      y: rightIris.y - rightEyeCenter.y,
      z: rightIris.z - rightEyeCenter.z
    };
    
    const avgGazeVec = {
      x: (leftGazeVec.x + rightGazeVec.x) / 2,
      y: (leftGazeVec.y + rightGazeVec.y) / 2,
      z: (leftGazeVec.z + rightGazeVec.z) / 2
    };
    
    const gazeYaw = Math.atan2(avgGazeVec.x, -avgGazeVec.z);
    const gazePitch = Math.atan2(avgGazeVec.y, -avgGazeVec.z);
    
    return {
      yaw: gazeYaw,
      pitch: gazePitch
    };
  };

  // Load leaderboard from storage on mount
  useEffect(() => {
    const loadLeaderboard = async () => {
      try {
        const result = await window.storage.get('emotional-charades-leaderboard');
        if (result && result.value) {
          setLeaderboard(JSON.parse(result.value));
        }
      } catch (err) {
        console.log('No existing leaderboard found');
        setLeaderboard([]);
      }
    };
    loadLeaderboard();
  }, []);

  // Save leaderboard to storage
  const saveLeaderboard = async (newLeaderboard) => {
    try {
      await window.storage.set('emotional-charades-leaderboard', JSON.stringify(newLeaderboard));
    } catch (err) {
      console.error('Failed to save leaderboard:', err);
    }
  };

  // Add score to leaderboard
  const addToLeaderboard = (name) => {
    const newEntry = {
      name: name || 'Anonymous',
      score: score,
      round: round - 1,
      date: new Date().toLocaleDateString(),
      difficulty: difficulty
    };
    
    const updatedLeaderboard = [...leaderboard, newEntry]
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
    
    setLeaderboard(updatedLeaderboard);
    saveLeaderboard(updatedLeaderboard);
    setShowNameInput(false);
    setPlayerName('');
  };

// load models (async)
useEffect(() => {
  const loadModels = async () => {
    try {
      console.log("Fetching model size...");

      const response = await fetch("/emotion_model.onnx");
      const blob = await response.blob();
      console.log("Model size:", (blob.size / 1024 / 1024).toFixed(2), "MB");

      console.log('Loading ONNX model...');
      let session;
      try {
        session = await ort.InferenceSession.create('/emotion_model.onnx');
      } catch (onnxErr) {
        console.log('ONNX model failed to load.')
      }
      sessionRef.current = session;
      console.log('ONNX model loaded successfully');
   
      console.log('Loading MediaPipe Face Landmarker...');
      const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3');
      const { FaceLandmarker, FilesetResolver } = vision;
      
      const filesetResolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        numFaces: 1,
        runningMode: 'VIDEO'
      });
      faceLandmarkerRef.current = landmarker;
      console.log('MediaPipe Face Landmarker loaded successfully');
      
      setModelLoaded(true);
    } catch (err) {
      console.error('Failed to load models:', err);
      setModelError(`Loading error: ${err.message}`);
    }
  };
  loadModels();
}, []);

// timer 
  useEffect(() => {
    if (gameState !== 'playing') return;
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          endGame();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [gameState]);

// start the camera 
  useEffect(() => {
    const startCamera = async () => {
      if (gameState === 'playing' && videoRef.current && modelLoaded) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 640 }, height: { ideal: 480 } } 
          });
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        } catch (err) {
          console.error("Camera error:", err);
          setModelError('Failed to access camera. Please check permissions.');
        }
      }
    };
    startCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [gameState, modelLoaded]);

  const getEulerAngles = (matrix) => {
    try {
      let matrixData;
      
      if (matrix.data) {
        matrixData = Array.from(matrix.data);
      } else if (Array.isArray(matrix)) {
        matrixData = matrix;
      } else {
        console.warn('Unknown matrix format:', matrix);
        return { yaw: 0, pitch: 0, roll: 0 };
      }
      
      const m = [
        [matrixData[0], matrixData[4], matrixData[8], matrixData[12]],
        [matrixData[1], matrixData[5], matrixData[9], matrixData[13]],
        [matrixData[2], matrixData[6], matrixData[10], matrixData[14]],
        [matrixData[3], matrixData[7], matrixData[11], matrixData[15]]
      ];
      
      let pitch = Math.asin(Math.max(-1, Math.min(1, -m[2][0])));
      let yaw = Math.atan2(m[1][0], m[0][0]);
      let roll = Math.atan2(m[2][1], m[2][2]);
      
      return {
        yaw: (yaw * 180) / Math.PI,
        pitch: (pitch * 180) / Math.PI,
        roll: (roll * 180) / Math.PI
      };
    } catch (err) {
      console.error('Error extracting Euler angles:', err);
      return { yaw: 0, pitch: 0, roll: 0 };
    }
  };

  // emotion predictions
  useEffect(() => {
    if (gameState !== 'playing' || !modelLoaded || !sessionRef.current || !videoRef.current || !faceLandmarkerRef.current) return;

    const interval = setInterval(async () => {
      try {
        if (videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA) return;

        const results = faceLandmarkerRef.current.detectForVideo(videoRef.current, Date.now());
        if (!results.facialTransformationMatrixes || results.facialTransformationMatrixes.length === 0) {
          console.log("No face detected.")
          return; 
        }
        
        const features = [];

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          const gazeAngles = estimateOpenFaceGaze(results.faceLandmarks[0]);
          features.push(gazeAngles.yaw, gazeAngles.pitch);
        } else {
          features.push(0, 0);
        }  
        
        const poseAngles = getEulerAngles(results.facialTransformationMatrixes[0]);
        features.push(poseAngles.yaw, poseAngles.pitch, poseAngles.roll);

        if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
          const blendshapes = results.faceBlendshapes[0];
          const blendshapeMap = {};
          
          blendshapes.categories.forEach(cat => {
            blendshapeMap[cat.categoryName] = cat.score;
          });

          KEY_BLENDSHAPES.forEach(name => {
            features.push(blendshapeMap[name] || 0);
          });
        } else {
          features.push(...Array(20).fill(0));
        }

        const scaledFeatures = standardizeFeatures(features);

        const inputTensor = new ort.Tensor('float32', new Float32Array(scaledFeatures), [1, features.length]);
        const feeds = { [sessionRef.current.inputNames[0]]: inputTensor };
        
        let emoResults;
        try {
          const options = { outputNames: ['probabilities'] };
          emoResults = await sessionRef.current.run(feeds, options);
        } catch (selectiveError) {
          console.log('Model prediction failed.');
        }      
        
        const probOutput = emoResults['probabilities'];
        
        let confidences;
        try {
          if (probOutput.cpuData) {
            confidences = Array.from(probOutput.cpuData);
          } else if (probOutput.data) {
            confidences = Array.from(probOutput.data);
          } else {
            console.error('Cannot find data in output');
            return;
          }
        } catch (dataError) {
          console.error('Error accessing tensor data:', dataError);
          return;
        }

        const newExpressions = emotionLabels.map((emotion, idx) => ({
          emotion,
          confidence: confidences[idx] || 0
        }));
        
        setDetectedExpressions(newExpressions);

        newExpressions.forEach(exp => {
          if (exp.confidence > bestEmotionRef.current.confidence) {
            bestEmotionRef.current = { emotion: exp.emotion, confidence: exp.confidence };
          }
        });

        // Windowing logic for consistent detection
        const targetExp = newExpressions.find(e => e.emotion === targetEmotion);
        
        // Add current detection to window
        if (targetExp && targetExp.confidence > MATCH_THRESHOLD) {
          detectionWindow.current.push(true);
        } else {
          detectionWindow.current.push(false);
        }

        // Keep window at fixed size
        if (detectionWindow.current.length > WINDOW_SIZE) {
          detectionWindow.current.shift();
        }

        // Calculate progress
        const matchesInWindow = detectionWindow.current.filter(x => x).length;
        const progress = (matchesInWindow / WINDOW_SIZE) * 100;
        setWindowProgress(progress);

        // Check if we have enough consistent detections
        if (detectionWindow.current.length === WINDOW_SIZE) {
          const consistencyRate = matchesInWindow / WINDOW_SIZE;
          
          if (consistencyRate >= CONSISTENCY_THRESHOLD) {
            // Award points based on average confidence
            const avgConfidence = targetExp ? targetExp.confidence : 0.7;
            const points = Math.round(avgConfidence * 100);
            setScore(prev => prev + points);
            
            // Track stats
            confidenceSum.current += avgConfidence;
            matchCount.current += 1;
            
            // Reset window and get new emotion
            detectionWindow.current = [];
            setWindowProgress(0);
            setTargetEmotion(getRandomEmotion());
          }
        }
      } catch (err) {
        console.error('Inference error:', err);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [gameState, modelLoaded, targetEmotion, KEY_BLENDSHAPES]);

  const startGame = () => {
      setCameraActive(true);
      setGameState('playing');
      setTimeLeft(60);
      setScore(0);
      setRound(1);
      confidenceSum.current = 0;
      bestEmotionRef.current = { emotion: '', confidence: 0 };
      matchCount.current = 0;
      detectionWindow.current = [];
      setWindowProgress(0);
      setTargetEmotion(getRandomEmotion());
  };

  const endGame = () => {
    const finalStats = {
      emotionsMatched: matchCount.current,
      totalAttempts: round - 1,
      averageConfidence: matchCount.current > 0 ? 
        Math.round((confidenceSum.current / matchCount.current) * 100) : 0,
      timeUsed: 60 - timeLeft,
      bestEmotion: bestEmotionRef.current.emotion || ''
    };
    setRoundStats(finalStats);
    
    // Check if score qualifies for leaderboard
    if (leaderboard.length < 10 || score > leaderboard[leaderboard.length - 1]?.score) {
      setShowNameInput(true);
    }
    
    setGameState('stats');
  };

  const pauseGame = () => {
    setGameState(gameState === 'playing' ? 'paused' : 'playing');
  };

  const resetGame = () => {
    setGameState('menu');
    setCameraActive(false);
    setScore(0);
    setTimeLeft(60);
    setShowNameInput(false);
  };

  const getRandomEmotion = () => {
    let emotions;
    if (difficulty === 'easy') {
      emotions = ['Happy', 'Sad', 'Fear'];
    } else if (difficulty === 'medium') {
      emotions = ['Happy', 'Sad', 'Fear', 'Surprised', 'Neutral'];
    } else {
      emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprised', 'Neutral', 'Disgust'];
    }
    return emotions[Math.floor(Math.random() * emotions.length)];
  };

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

  if (modelError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-6 flex items-center justify-center">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold text-red-400">Error</h2>
          <p>{modelError}</p>
          <p className="text-sm text-gray-300">Make sure model.onnx is in your public folder</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-purple-700 to-blue-800 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6 bg-white bg-opacity-10 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <div className="text-4xl">🎭</div>
            <div>
              <h1 className="text-3xl font-bold">Emotional Charades</h1>
              <p className="text-sm text-gray-300">AI-Powered Expression Game</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {!modelLoaded && (
              <div className="flex items-center gap-2 bg-yellow-500 bg-opacity-20 px-3 py-2 rounded-full">
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                <span className="text-sm">Loading...</span>
              </div>
            )}
            {modelLoaded && (
              <div className="flex items-center gap-2 bg-green-500 bg-opacity-20 px-3 py-2 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm">Ready</span>
              </div>
            )}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-white hover:bg-opacity-10 rounded-lg transition-all"
            >
              <Settings size={24} />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-6 bg-white bg-opacity-10 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Settings size={20} />
              Game Settings
            </h2>
            <div className="max-w-md">
              <label className="block text-sm font-medium mb-2">Difficulty Level</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                className="bg-white bg-opacity-20 rounded-lg px-4 py-2 w-full text-white"
              >
                <option value="easy" className="bg-gray-800">Easy - 3 Emotions</option>
                <option value="medium" className="bg-gray-800">Medium - 5 Emotions</option>
                <option value="hard" className="bg-gray-800">Hard - 7 Emotions</option>
              </select>
              <p className="text-xs text-gray-300 mt-2">
                {difficulty === 'easy' && '😊 😢 😬'}
                {difficulty === 'medium' && '😊 😢 😬 😮 😐'}
                {difficulty === 'hard' && '😊 😢 😠 😬 😮 😐 🤮'}
              </p>
            </div>
          </div>
        )}

        {/* Stats Screen */}
        {gameState === 'stats' && (
          <div className="space-y-6">
            <div className="text-center space-y-4">
              <div className="text-6xl">🎉</div>
              <h2 className="text-4xl font-bold">Round Complete!</h2>
              <p className="text-xl text-gray-300">Here's how you did</p>
            </div>

            {/* Final Score */}
            <div className="bg-gradient-to-r from-pink-500 to-purple-500 bg-opacity-30 rounded-lg p-8 text-center">
              <p className="text-sm font-medium mb-2">FINAL SCORE</p>
              <p className="text-7xl font-bold">{score}</p>
            </div>

            {/* Statistics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white bg-opacity-10 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-300 mb-1">Emotions Matched</p>
                <p className="text-3xl font-bold">{roundStats.emotionsMatched}</p>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-300 mb-1">Rounds Played</p>
                <p className="text-3xl font-bold">{round}</p>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-300 mb-1">Avg Confidence</p>
                <p className="text-3xl font-bold">{roundStats.averageConfidence}%</p>
              </div>
              <div className="bg-white bg-opacity-10 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-300 mb-1">Best Emotion</p>
                <p className="text-3xl">{emojiDictionary(roundStats.bestEmotion)}</p>
              </div>
            </div>

            {/* Name Input for Leaderboard */}
            {showNameInput && (
              <div className="bg-yellow-500 bg-opacity-20 rounded-lg p-6 text-center space-y-4">
                <Trophy size={48} className="mx-auto text-yellow-400" />
                <h3 className="text-2xl font-bold">You made the leaderboard!</h3>
                <p className="text-gray-300">Enter your name to save your score</p>
                <div className="flex gap-2 max-w-md mx-auto">
                  <input
                    type="text"
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value)}
                    placeholder="Your name"
                    maxLength={20}
                    className="flex-1 px-4 py-2 rounded-lg bg-white bg-opacity-20 text-white placeholder-gray-400"
                    onKeyPress={(e) => e.key === 'Enter' && addToLeaderboard(playerName)}
                  />
                  <button
                    onClick={() => addToLeaderboard(playerName)}
                    className="bg-green-500 hover:bg-green-600 px-6 py-2 rounded-lg font-bold transition-all"
                  >
                    Save
                  </button>
                </div>
              </div>
            )}


            {/* Leaderboard */}
            <div className="bg-white bg-opacity-10 rounded-lg p-6">
              <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Trophy className="text-yellow-400" />
                Leaderboard
              </h3>
              {leaderboard.length === 0 ? (
                <p className="text-center text-gray-300 py-8">No scores yet. Be the first!</p>
              ) : (
                <div className="space-y-2">
                  {leaderboard.map((entry, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center justify-between p-3 rounded-lg ${
                        idx === 0 ? 'bg-yellow-500 bg-opacity-30' :
                        idx === 1 ? 'bg-gray-400 bg-opacity-20' :
                        idx === 2 ? 'bg-orange-600 bg-opacity-20' :
                        'bg-white bg-opacity-5'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-2xl font-bold w-8">
                          {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `${idx + 1}.`}
                        </span>
                        <div>
                          <p className={
                            entry.difficulty === 'easy' ? 'text-green-400' :
                            entry.difficulty === 'medium' ? 'text-yellow-400' :
                            'text-red-400'
                          }>
                            {entry.difficulty.toUpperCase()}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold">{entry.score}</p>
                        <p className="text-sm text-gray-300">{entry.round} rounds</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex justify-center gap-4">
              <button
                onClick={() => {
                  setRound(prev => prev + 1);  // ⭐ increment round here
                  startGame();
                }}
              >
                Play Again
              </button>
              <button
                onClick={startGame}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 px-8 py-4 rounded-lg text-xl font-bold transition-all"
              >
                <Play size={24} />
                Play Again
              </button>
              <button
                onClick={resetGame}
                className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 px-8 py-4 rounded-lg text-xl font-bold transition-all"
              >
                <Home size={24} />
                Main Menu
              </button>
            </div>
          </div>
        )}

        {/* Menu Screen */}
        {gameState === 'menu' && (
          <div className="text-center space-y-8 py-12">
            <div className="text-8xl mb-4">🎭</div>
            
            <div className="space-y-4">
              <h2 className="text-4xl font-bold">Ready to Express Yourself?</h2>
              <p className="text-xl text-gray-300 max-w-2xl mx-auto">
                Match the target emotions using your facial expressions. The AI will detect and score your performance!
              </p>
            </div>

            <button
              onClick={startGame}
              disabled={!modelLoaded}
              className="bg-pink-500 hover:bg-pink-600 px-10 py-4 rounded-lg text-xl font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-3"
            >
              <Play size={24} />
              {modelLoaded ? 'Start Playing' : 'Loading Models...'}
            </button>
            
            <div className="flex justify-center gap-6 mt-8">
              {['😊', '😢', '😠', '😮', '😐', '😨'].map((emoji, i) => (
                <div key={i} className="text-4xl opacity-50 hover:opacity-100 transition-opacity">
                  {emoji}
                </div>
              ))}
            </div>

            {/* Leaderboard Preview on Menu */}
            {leaderboard.length > 0 && (
              <div className="mt-12 max-w-2xl mx-auto bg-white bg-opacity-10 rounded-lg p-6">
                <h3 className="text-2xl font-bold mb-4 flex items-center justify-center gap-2">
                  <Trophy className="text-yellow-400" />
                  Top Scores
                </h3>
                <div className="space-y-2">
                  {leaderboard.slice(0, 5).map((entry, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-2 rounded-lg bg-white bg-opacity-5"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xl font-bold w-6">
                          {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `${idx + 1}.`}
                        </span>
                        <span className="font-medium">{entry.name}</span>
                      </div>
                      <span className="text-xl font-bold">{entry.score}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Playing Screen */}
        {(gameState === 'playing' || gameState === 'paused') && (
          <div className="space-y-6">
            {/* Stats Dashboard */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-pink-500 bg-opacity-20 rounded-lg p-4 text-center">
                <p className="text-sm font-medium mb-1">Score</p>
                <p className="text-4xl font-bold">{score}</p>
              </div>
              <div className="bg-purple-500 bg-opacity-20 rounded-lg p-4 text-center">
                <p className="text-sm font-medium mb-1">Round</p>
                <p className="text-4xl font-bold">{round}</p>
              </div>
              <div className="bg-yellow-500 bg-opacity-20 rounded-lg p-4 text-center">
                <p className="text-sm font-medium mb-1">Time</p>
                <p className={`text-4xl font-bold ${timeLeft < 10 ? 'text-red-400' : ''}`}>
                  {timeLeft}s
                </p>
              </div>
            </div>

            {/* Target Emotion */}
            <div className="bg-gradient-to-r from-pink-500 to-purple-500 bg-opacity-20 rounded-lg p-6 text-center">
              <p className="text-sm font-medium mb-2">TARGET EMOTION</p>
              <div className="text-7xl mb-3">{emojiDictionary(targetEmotion)}</div>
              <p className="text-3xl font-bold">{targetEmotion}</p>
              
              {/* Window Progress Bar */}
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Detection Progress</span>
                  <span className="font-bold">{Math.round(windowProgress)}%</span>
                </div>
                <div className="bg-white bg-opacity-20 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      windowProgress >= 70 ? 'bg-green-500' :
                      windowProgress >= 40 ? 'bg-yellow-400' :
                      'bg-red-400'
                    }`}
                    style={{ width: `${windowProgress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-300">
                  Hold the expression consistently to score!
                </p>
              </div>
            </div>

            {/* Video and Detection */}
            <div className="grid md:grid-cols-3 gap-4">
              {/* Video Feed */}
              <div className="md:col-span-2">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-3 left-3 bg-red-500 px-3 py-1 rounded-full flex items-center gap-2">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                    <span className="text-sm font-bold">LIVE</span>
                  </div>
                  {gameState === 'paused' && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                      <div className="text-center">
                        <Pause size={64} className="mx-auto mb-4" />
                        <p className="text-2xl font-bold">PAUSED</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Emotion Detection Panel */}
              <div className="bg-white bg-opacity-10 rounded-lg p-4 space-y-3">
                <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  AI Detection
                </h3>
                {detectedExpressions.map((exp) => (
                  <div key={exp.emotion} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span>{emojiDictionary(exp.emotion)}</span>
                        <span className="text-sm font-medium">{exp.emotion}</span>
                      </div>
                      <span className="text-sm font-bold">
                        {Math.round(exp.confidence * 100)}%
                      </span>
                    </div>
                    <div className="bg-white bg-opacity-20 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full transition-all ${getConfidenceColor(exp.confidence)}`}
                        style={{ width: `${exp.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Control Buttons */}
            <div className="flex justify-center gap-4">
              <button
                onClick={pauseGame}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-bold transition-all"
              >
                {gameState === 'playing' ? <Pause size={20} /> : <Play size={20} />}
                {gameState === 'playing' ? 'Pause' : 'Resume'}
              </button>
              <button
                onClick={resetGame}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg font-bold transition-all"
              >
                <RotateCcw size={20} />
                Exit Game
              </button>
            </div>
          </div>
        )}
      </div>
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}