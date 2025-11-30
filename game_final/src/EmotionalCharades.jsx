// use react + lucide-react + onnx runtime web lib
import React, { useState, useRef, useEffect } from 'react';
import { Settings, Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import * as ort from 'onnxruntime-web';

export default function EmotionalCharades() {

  // game flow 
  const [gameState, setGameState] = useState('menu'); // menu, playing, paused
  const [score, setScore] = useState(0);
  const [round, setRound] = useState(1);
  const [timeLeft, setTimeLeft] = useState(60);

  // ui settings
  const [showSettings, setShowSettings] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false); // sound off, can be improved later
  const [difficulty, setDifficulty] = useState('medium'); // default

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

  // emotion labels (default emotion: happy)
  const emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'];
  const [targetEmotion, setTargetEmotion] = useState('Happy');

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
    // MediaPipe iris landmarks (these track pupil position)
    const LEFT_IRIS_CENTER = 468;
    const RIGHT_IRIS_CENTER = 473;
    
    // Eye corner landmarks (to get eye center)
    const LEFT_EYE_INNER = 133;
    const LEFT_EYE_OUTER = 33;
    const RIGHT_EYE_INNER = 362;
    const RIGHT_EYE_OUTER = 263;
    
    // Calculate eye centers
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
    
    // Get iris centers
    const leftIris = faceLandmarks[LEFT_IRIS_CENTER];
    const rightIris = faceLandmarks[RIGHT_IRIS_CENTER];
    
    // Calculate gaze vectors (iris position relative to eye center)
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
    
    // Average both eyes (OpenFace typically does this)
    const avgGazeVec = {
      x: (leftGazeVec.x + rightGazeVec.x) / 2,
      y: (leftGazeVec.y + rightGazeVec.y) / 2,
      z: (leftGazeVec.z + rightGazeVec.z) / 2
    };
    
    // Convert to OpenFace-style angles
    // Note: MediaPipe uses different coordinate system, may need sign adjustments
    const gazeYaw = Math.atan2(avgGazeVec.x, -avgGazeVec.z);
    const gazePitch = Math.atan2(avgGazeVec.y, -avgGazeVec.z);
    
    return {
      yaw: gazeYaw,    // Already in radians
      pitch: gazePitch // Already in radians
    };
  };

// load models (async)
useEffect(() => {
  const loadModels = async () => {
    try {
      // load ONNX emotion model
      console.log('Loading ONNX model...');
      let session;
      try {
        session = await ort.InferenceSession.create('/emotion_model.onnx');
      } catch (onnxErr) {
        console.log('ONNX model failed to load.')
      }
      sessionRef.current = session;
      console.log('ONNX model loaded successfully');
   
      // load MediaPipe Face Landmarker (dynamic import)
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
          setGameState('menu');
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

// get gaze yaw and pitch
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
      
      // matrix is in column-major order (OpenGL style)
      // convert to row-major for easier access: m[row][col]
      const m = [
        [matrixData[0], matrixData[4], matrixData[8], matrixData[12]],
        [matrixData[1], matrixData[5], matrixData[9], matrixData[13]],
        [matrixData[2], matrixData[6], matrixData[10], matrixData[14]],
        [matrixData[3], matrixData[7], matrixData[11], matrixData[15]]
      ];
      
      // rotation angles
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
          console.log("No face detected.") // add user warning 
          return; 
        }
        
        const features = [];

        // gaze features 
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
          const gazeAngles = estimateOpenFaceGaze(results.faceLandmarks[0]);
          features.push(gazeAngles.yaw, gazeAngles.pitch);
          console.log(`Gaze - Yaw: ${gazeAngles.yaw.toFixed(3)}, Pitch: ${gazeAngles.pitch.toFixed(3)}`);  
        } else {
          // Fallback to zeros if no landmarks
          features.push(0, 0);
        }  
        
        // pose features
        const poseAngles = getEulerAngles(results.facialTransformationMatrixes[0]);
        features.push(poseAngles.yaw, poseAngles.pitch, poseAngles.roll);

        // key blend shapes
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
          // user warning
          features.push(...Array(20).fill(0));
        }

        const scaledFeatures = standardizeFeatures(features);

        const inputTensor = new ort.Tensor('float32', new Float32Array(scaledFeatures), [1, features.length]);
        const feeds = { [sessionRef.current.inputNames[0]]: inputTensor };
        
        let emoResults;
        try {
          const options = { outputNames: ['probabilities'] };
          emoResults = await sessionRef.current.run(feeds, options);
          console.log('Model prediction successful.');
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
          console.log('Got confidences:', confidences);
        } catch (dataError) {
          console.error('Error accessing tensor data:', dataError);
          return;
        }

        const newExpressions = emotionLabels.map((emotion, idx) => ({
          emotion,
          confidence: confidences[idx] || 0
        }));
        
        setDetectedExpressions(newExpressions);

        // check if target emotion is detected
        const targetExp = newExpressions.find(e => e.emotion === targetEmotion);
        if (targetExp && targetExp.confidence > 0.6) {
          setScore(prev => prev + Math.round(targetExp.confidence * 100));
          setRound(prev => prev + 1);
          setTargetEmotion(getRandomEmotion());
        }
      } catch (err) {
        console.error('Inference error:', err);
      }
    }, 200);

    return () => clearInterval(interval);
  }, [gameState, modelLoaded, targetEmotion, KEY_BLENDSHAPES]);

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
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Difficulty Level</label>
                <select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  className="bg-white bg-opacity-20 rounded-lg px-4 py-2 w-full text-white"
                >
                  <option value="easy" className="bg-gray-800">Easy - 70s</option>
                  <option value="medium" className="bg-gray-800">Medium - 60s</option>
                  <option value="hard" className="bg-gray-800">Hard - 45s</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Audio</label>
                <button
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  className="flex items-center gap-2 p-2 hover:bg-white hover:bg-opacity-10 rounded-lg transition-all w-full justify-center"
                >
                  {soundEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
                  <span>{soundEnabled ? 'Sound On' : 'Sound Off'}</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Menu Screen */}
        {gameState === 'menu' ? (
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
          </div>
        ) : (
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