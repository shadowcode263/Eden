"use client"

import type React from "react"

import { useEffect, useRef, useState, useCallback } from "react"
import { Canvas, useThree, useFrame } from "@react-three/fiber"
import { OrbitControls, Html } from "@react-three/drei"
import * as THREE from "three"
import { useWebSocket } from "@/hooks/use-websocket"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, Wifi, WifiOff } from "lucide-react"

interface BrainVisualizationProps {
  networkId: number
  embeddingDim: number // New prop: embedding dimension from the network
  onRetrievalSummaryUpdate: (data: RetrievalSummaryData) => void // Callback to update parent
}

interface NetworkUpdate {
  type: string
  event_type: string
  data: any
}

interface RetrievalSummaryData {
  query_text: string
  confidence_score: number
  steps_count: number
  retrieved_hash: string
  retrieved_text: string
}

// Define NeuronAnimationState type
interface NeuronAnimationState {
  targetScale: number
  targetPosition: THREE.Vector3
  targetColor: THREE.Color
  targetEmissive: THREE.Color
}

// New component for the 3D scene content and HTML overlays
function BrainSceneContent({
  embeddingDim,
  lastMessage,
  activity,
  setActivity,
  localRetrievalSummaryData,
  setLocalRetrievalSummaryData,
  isConnected,
  connectionHealth,
  connectionStatus,
  getConnectionStatusColor,
  getConnectionStatusText,
}: {
  embeddingDim: number
  lastMessage: any
  activity: string
  setActivity: React.Dispatch<React.SetStateAction<string>>
  localRetrievalSummaryData: RetrievalSummaryData | null
  setLocalRetrievalSummaryData: React.Dispatch<React.SetStateAction<RetrievalSummaryData | null>>
  isConnected: boolean
  connectionHealth: any
  connectionStatus: string
  getConnectionStatusColor: () => string
  getConnectionStatusText: () => string
}) {
  const { scene, camera, gl } = useThree()
  const controlsRef = useRef<any>() // OrbitControls ref
  const neuronsRef = useRef<THREE.Mesh[]>([])
  // Ref to store connection data: { line object, index of neuron1, index of neuron2 }
  const connectionDataRef = useRef<{ line: THREE.Line; startNeuronIndex: number; endNeuronIndex: number }[]>([])
  // New useRef for managing neuron animation states
  const neuronAnimationStateRef = useRef(new Map<number, NeuronAnimationState>())
  // Ref to store setTimeout IDs for cleanup
  const retrievalAnimationTimeoutsRef = useRef<NodeJS.Timeout[]>([])

  // Three.js setup
  useEffect(() => {
    // Cleanup function for previous Three.js instance
    const cleanupThree = () => {
      // Clear any pending animation timeouts
      retrievalAnimationTimeoutsRef.current.forEach(clearTimeout)
      retrievalAnimationTimeoutsRef.current = []

      // Dispose of geometries and materials
      scene.children.forEach((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          scene.remove(obj)
          obj.geometry.dispose()
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose())
          } else if (obj.material) {
            obj.material.dispose()
          }
        }
      })
      // Clear refs
      neuronsRef.current = []
      connectionDataRef.current = []
      neuronAnimationStateRef.current.clear() // Clear animation state map
    }

    // Perform cleanup before re-initializing
    cleanupThree()

    // Basic scene setup (background, fog, lights)
    scene.background = new THREE.Color(0x0a0a0f)
    scene.fog = new THREE.Fog(0x0a0a0f, 10, 50)

    const ambientLight = new THREE.AmbientLight(0x1a1a2e, 0.3)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0x4a9eff, 1.2)
    directionalLight.position.set(10, 10, 5)
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    scene.add(directionalLight)

    const directionalLight2 = new THREE.DirectionalLight(0x9d4edd, 0.8)
    directionalLight2.position.set(-10, -5, -5)
    scene.add(directionalLight2)

    const pointLight = new THREE.PointLight(0x00ffff, 0.6, 20)
    pointLight.position.set(0, 8, 0)
    scene.add(pointLight)

    // Initial camera position
    camera.position.set(0, 5, 18)
    camera.lookAt(0, 0, 0)

    // Create brain structure (neurons and connections)
    createFuturisticBrainStructure(scene, embeddingDim, neuronsRef, connectionDataRef, neuronAnimationStateRef)

    // Return cleanup function for this effect
    return () => {
      cleanupThree() // Ensure cleanup on unmount or embeddingDim changes
    }
  }, [scene, camera, embeddingDim]) // Dependencies for Three.js setup

  // Animation loop using useFrame
  useFrame(() => {
    if (controlsRef.current) {
      controlsRef.current.update()
    }

    // Apply neuron animation states
    neuronsRef.current.forEach((neuron, index) => {
      const state = neuronAnimationStateRef.current.get(index)
      if (state) {
        const material = neuron.material as THREE.MeshLambertMaterial
        const lerpFactor = 0.1 // Smoothness factor

        // Lerp position
        neuron.position.lerp(state.targetPosition, lerpFactor)

        // Lerp scale
        neuron.scale.lerp(new THREE.Vector3(state.targetScale, state.targetScale, state.targetScale), lerpFactor)

        // Lerp color and emissive
        material.color.lerp(state.targetColor, lerpFactor)
        material.emissive.lerp(state.targetEmissive, lerpFactor)
      }
    })

    // Update connection positions based on current neuron positions
    connectionDataRef.current.forEach(({ line, startNeuronIndex, endNeuronIndex }) => {
      const startNeuron = neuronsRef.current[startNeuronIndex]
      const endNeuron = neuronsRef.current[endNeuronIndex]

      if (startNeuron && endNeuron) {
        const positions = line.geometry.attributes.position.array as Float32Array
        positions[0] = startNeuron.position.x
        positions[1] = startNeuron.position.y
        positions[2] = startNeuron.position.z
        positions[3] = endNeuron.position.x
        positions[4] = endNeuron.position.y
        positions[5] = endNeuron.position.z
        line.geometry.attributes.position.needsUpdate = true // Important!
      }
    })

    gl.render(scene, camera) // Explicitly render
  })

  // Network update handling
  useEffect(() => {
    if (lastMessage) {
      handleNetworkUpdate(lastMessage)
    }
  }, [lastMessage])

  // Brain control event handler
  useEffect(() => {
    const handleBrainControl = (event: CustomEvent) => {
      const { type, value } = event.detail
      if (!controlsRef.current) return

      switch (type) {
        case "zoom":
          const currentPosition = camera.position.clone()
          const direction = currentPosition.clone().normalize()
          camera.position.copy(direction.multiplyScalar(value))
          controlsRef.current.target.set(0, 0, 0)
          controlsRef.current.update()
          break
        case "rotationSpeed":
          controlsRef.current.autoRotateSpeed = value
          break
        case "autoRotate":
          controlsRef.current.autoRotate = value
          break
        case "reset":
          camera.position.set(0, 5, 18)
          camera.lookAt(0, 0, 0)
          controlsRef.current.target.set(0, 0, 0)
          controlsRef.current.autoRotateSpeed = 1.0
          controlsRef.current.autoRotate = true
          controlsRef.current.update()
          break
      }
    }

    window.addEventListener("brainControl", handleBrainControl as EventListener)
    return () => {
      window.removeEventListener("brainControl", handleBrainControl as EventListener)
    }
  }, [camera]) // Dependency on camera for position/lookAt

  const createFuturisticBrainStructure = (
    scene: THREE.Scene,
    numNeurons: number,
    neuronsRef: React.MutableRefObject<THREE.Mesh[]>,
    connectionDataRef: React.MutableRefObject<{ line: THREE.Line; startNeuronIndex: number; endNeuronIndex: number }[]>,
    neuronAnimationStateRef: React.MutableRefObject<Map<number, NeuronAnimationState>>,
  ) => {
    // Clear existing objects
    scene.children.forEach((obj) => {
      if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
        scene.remove(obj)
        obj.geometry.dispose()
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => m.dispose())
        } else if (obj.material) {
          obj.material.dispose()
        }
      }
    })
    neuronsRef.current = []
    connectionDataRef.current = []
    neuronAnimationStateRef.current.clear() // Clear animation state map

    const neuronGeometry = new THREE.SphereGeometry(0.08, 16, 16)
    const neurons: THREE.Mesh[] = []

    const phiIncrement = Math.PI * (3 - Math.sqrt(5))
    const sphereRadius = 6.5 // Radius for neuron placement

    const defaultNeuronColor = new THREE.Color(0x007bff)
    const defaultNeuronEmissive = new THREE.Color(0x002244)

    for (let i = 0; i < numNeurons; i++) {
      const y = 1 - (i / (numNeurons - 1)) * 2
      const radiusAtY = Math.sqrt(1 - y * y)

      const theta = i * phiIncrement

      const neuronMaterial = new THREE.MeshLambertMaterial({
        color: defaultNeuronColor,
        transparent: true,
        opacity: 0.9,
        emissive: defaultNeuronEmissive,
      })
      const neuron = new THREE.Mesh(neuronGeometry, neuronMaterial)

      neuron.position.x = radiusAtY * Math.cos(theta) * sphereRadius
      neuron.position.y = y * sphereRadius
      neuron.position.z = radiusAtY * Math.sin(theta) * sphereRadius

      // Store the original position for later reference
      neuron.userData.originalPosition = neuron.position.clone()

      neuron.castShadow = true
      neuron.receiveShadow = true
      scene.add(neuron)
      neurons.push(neuron)

      // Initialize animation state for each neuron
      neuronAnimationStateRef.current.set(i, {
        targetScale: 1,
        targetPosition: neuron.position.clone(),
        targetColor: defaultNeuronColor.clone(),
        targetEmissive: defaultNeuronEmissive.clone(),
      })
    }

    neuronsRef.current = neurons

    const connectionsData: { line: THREE.Line; startNeuronIndex: number; endNeuronIndex: number }[] = []
    // Make connections much denser for smaller networks
    if (numNeurons <= 200) {
      // Apply denser connections up to 200 neurons
      for (let i = 0; i < neurons.length; i++) {
        for (let j = i + 1; j < neurons.length; j++) {
          const distance = neurons[i].position.distanceTo(neurons[j].position)
          // Connect if within a larger visual proximity
          if (distance < 10.0) {
            // Increased distance threshold for denser connections
            const geometry = new THREE.BufferGeometry()
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3)) // 2 vertices * 3 components
            const material = new THREE.LineBasicMaterial({
              color: 0xcccccc,
              transparent: true,
              opacity: 0.4,
            })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
            connectionsData.push({ line, startNeuronIndex: i, endNeuronIndex: j })
          }
        }
      }
    } else {
      // For very large networks (e.g., 512), keep sparse connections for performance
      for (let i = 0; i < neurons.length; i++) {
        for (let j = i + 1; j < neurons.length; j++) {
          const distance = neurons[i].position.distanceTo(neurons[j].position)
          if (distance < 3.0 && Math.random() < 0.05) {
            // Keep some sparse connections
            const geometry = new THREE.BufferGeometry()
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3)) // 2 vertices * 3 components
            const material = new THREE.LineBasicMaterial({
              color: 0xcccccc,
              transparent: true,
              opacity: 0.4,
            })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
            connectionsData.push({ line, startNeuronIndex: i, endNeuronIndex: j })
          }
        }
      }
    }
    connectionDataRef.current = connectionsData

    const particleGeometry = new THREE.BufferGeometry()
    const particleCount = 200
    const positions = new Float32Array(particleCount * 3)

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 20
      positions[i + 1] = (Math.random() - 0.5) * 20
      positions[i + 2] = (Math.random() - 0.5) * 20
    }

    particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))

    const particleMaterial = new THREE.PointsMaterial({
      color: 0x00ffff,
      size: 0.02,
      transparent: true,
      opacity: 0.6,
    })

    const particles = new THREE.Points(particleGeometry, particleMaterial)
    scene.add(particles)
  }

  const handleNetworkUpdate = (update: NetworkUpdate) => {
    switch (update.event_type) {
      case "pattern_stored":
        animatePatternStorage(update.data)
        break
      case "pattern_retrieved":
        animatePatternRetrieval(update.data)
        setLocalRetrievalSummaryData({
          query_text: update.data.query_text,
          confidence_score: update.data.confidence,
          steps_count: update.data.steps?.length || 0,
          retrieved_hash: update.data.retrieved_hash,
          retrieved_text: update.data.retrieved_text,
        })
        break
    }
  }

  const animatePatternStorage = (data: any) => {
    setActivity(`Storing Pattern: "${data.text}"`)
    setLocalRetrievalSummaryData(null)

    // Clear any existing retrieval animation timeouts
    retrievalAnimationTimeoutsRef.current.forEach(clearTimeout)
    retrievalAnimationTimeoutsRef.current = []

    const numActiveNeurons = Math.min(30, neuronsRef.current.length)
    const activeNeuronsIndices = neuronsRef.current
      .map((_, i) => i)
      .sort(() => 0.5 - Math.random())
      .slice(0, numActiveNeurons)

    // Reset all connections to default before animating
    connectionDataRef.current.forEach(({ line }) => {
      const connectionMaterial = line.material as THREE.LineBasicMaterial
      connectionMaterial.color.setHex(0xcccccc)
      connectionMaterial.opacity = 0.4
    })

    const storageExcitationDistance = 0.5 // How much neurons move outward during storage
    const storageVerticalPulseMagnitude = 0.2 // How much neurons pulse up/down during storage

    activeNeuronsIndices.forEach((neuronIndex, index) => {
      const timeoutId = setTimeout(() => {
        const neuron = neuronsRef.current[neuronIndex]
        if (neuron) {
          const originalPos = neuron.userData.originalPosition.clone()
          const direction = originalPos.clone().normalize()
          const pulseOffset = Math.sin(index * 0.5) * storageVerticalPulseMagnitude // Use index for varied pulse

          const targetPos = originalPos
            .clone()
            .add(direction.multiplyScalar(storageExcitationDistance))
            .add(new THREE.Vector3(0, pulseOffset, 0))

          // Update target state for this neuron
          neuronAnimationStateRef.current.set(neuronIndex, {
            targetScale: 1.2, // Slightly larger
            targetPosition: targetPos,
            targetColor: new THREE.Color(0x00ff00), // Green
            targetEmissive: new THREE.Color(0x004400),
          })
        }

        // Animate connections
        connectionDataRef.current.forEach(({ line }) => {
          const connectionMaterial = line.material as THREE.LineBasicMaterial
          connectionMaterial.color.setHex(0x00ff00) // Green
          connectionMaterial.opacity = 0.8
        })

        const resetTimeoutId = setTimeout(() => {
          // Reset target state for this neuron
          if (neuron) {
            neuronAnimationStateRef.current.set(neuronIndex, {
              targetScale: 1,
              targetPosition: neuron.userData.originalPosition.clone(),
              targetColor: new THREE.Color(0x007bff), // Default blue
              targetEmissive: new THREE.Color(0x002244),
            })
          }

          // Reset connections
          connectionDataRef.current.forEach(({ line }) => {
            const connectionMaterial = line.material as THREE.LineBasicMaterial
            connectionMaterial.color.setHex(0xcccccc) // Default gray
            connectionMaterial.opacity = 0.4
          })
        }, 1200) // Duration for the storage animation
        retrievalAnimationTimeoutsRef.current.push(resetTimeoutId)
      }, index * 30) // Staggered animation
      retrievalAnimationTimeoutsRef.current.push(timeoutId)
    })

    const finalActivityTimeoutId = setTimeout(() => setActivity("Pattern Stored Successfully"), 2500)
    retrievalAnimationTimeoutsRef.current.push(finalActivityTimeoutId)
  }

  const animatePatternRetrieval = (data: any) => {
    setActivity(`Retrieving: "${data.query_text}"`)

    // Clear any existing retrieval animation timeouts
    retrievalAnimationTimeoutsRef.current.forEach(clearTimeout)
    retrievalAnimationTimeoutsRef.current = []

    // Reset all neurons to dormant state by updating their target states
    neuronsRef.current.forEach((neuron, index) => {
      neuronAnimationStateRef.current.set(index, {
        targetScale: 1,
        targetPosition: neuron.userData.originalPosition.clone(),
        targetColor: new THREE.Color(0x007bff), // Default blue
        targetEmissive: new THREE.Color(0x002244),
      })
    })

    const firingThreshold = 0.001
    const excitationMultiplier = 4.0 // Outward movement
    const verticalPulseMagnitude = 0.5 // For up/down movement

    data.steps?.forEach((step: any, stepIndex: number) => {
      const timeoutId = setTimeout(() => {
        step.state?.forEach((stateValue: number, neuronIndex: number) => {
          if (neuronIndex < neuronsRef.current.length) {
            const neuron = neuronsRef.current[neuronIndex]
            const originalPos = neuron.userData.originalPosition.clone()

            if (stateValue > firingThreshold) {
              const direction = originalPos.clone().normalize()
              const excitationDistance = stateValue * excitationMultiplier

              // Calculate new target position: original + outward + vertical pulse
              const targetPos = originalPos
                .clone()
                .add(direction.multiplyScalar(excitationDistance))
                .add(new THREE.Vector3(0, Math.sin(stepIndex * 0.5) * verticalPulseMagnitude, 0)) // Simple sine wave for vertical pulse

              // Update target state for firing neuron
              neuronAnimationStateRef.current.set(neuronIndex, {
                targetScale: 1 + stateValue * 0.5,
                targetPosition: targetPos,
                targetColor: new THREE.Color(0x00ff00), // Green
                targetEmissive: new THREE.Color(0x004400),
              })
            } else {
              // Update target state for non-firing neuron (return to default)
              neuronAnimationStateRef.current.set(neuronIndex, {
                targetScale: 1,
                targetPosition: originalPos,
                targetColor: new THREE.Color(0x007bff), // Default blue
                targetEmissive: new THREE.Color(0x002244),
              })
            }
          }
        })

        // Animate connections for this step
        connectionDataRef.current.forEach(({ line, startNeuronIndex, endNeuronIndex }) => {
          const material = line.material as THREE.LineBasicMaterial
          // Check if either connected neuron is firing above threshold
          const startFiring = step.state[startNeuronIndex] > firingThreshold
          const endFiring = step.state[endNeuronIndex] > firingThreshold

          if (startFiring || endFiring) {
            material.color.setHex(0x00ff00) // Green
            material.opacity = 0.8
          } else {
            material.color.setHex(0xcccccc) // Default gray
            material.opacity = 0.4
          }
        })
      }, stepIndex * 120) // Delay per step
      retrievalAnimationTimeoutsRef.current.push(timeoutId)
    })

    // Final reset of neuron and connection target states after animation completes
    const finalResetTimeoutId = setTimeout(
      () => {
        setActivity(`Retrieved with ${(data.confidence * 100).toFixed(1)}% confidence`)
        neuronsRef.current.forEach((neuron, index) => {
          neuronAnimationStateRef.current.set(index, {
            targetScale: 1,
            targetPosition: neuron.userData.originalPosition.clone(),
            targetColor: new THREE.Color(0x007bff),
            targetEmissive: new THREE.Color(0x002244),
          })
        })
        connectionDataRef.current.forEach(({ line }) => {
          const material = line.material as THREE.LineBasicMaterial
          material.color.setHex(0xcccccc)
          material.opacity = 0.4
        })
      },
      (data.steps?.length || 0) * 120 + 2000, // Total animation time + 2 seconds for final state
    )
    retrievalAnimationTimeoutsRef.current.push(finalResetTimeoutId)
  }

  return (
    <>
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        enableZoom
        enableRotate
        enablePan
        autoRotate
        autoRotateSpeed={1.0}
        minDistance={8} // Adjusted minDistance for larger globe
        maxDistance={40} // Adjusted maxDistance for larger globe
      />

      {/* Combined Top-Left Overlay */}
      <Html fullscreen className="absolute top-4 left-4" zIndexRange={[100, 0]}>
        <div className="bg-black/80 backdrop-blur-sm text-cyan-100 px-4 py-3 rounded-lg shadow-lg border border-cyan-500/30">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
            <span className="text-xs font-mono">Neural Matrix Active</span>
          </div>
          <div className="flex items-center space-x-2">
            {isConnected ? <Wifi className="w-4 h-4 text-green-400" /> : <WifiOff className="w-4 h-4 text-red-400" />}
            <span className="text-xs text-cyan-300">{getConnectionStatusText()}</span>
            {connectionHealth.reconnectAttempts > 0 && (
              <Badge variant="outline" className="text-xs px-2 py-0 border-yellow-400 text-yellow-300">
                Retry {connectionHealth.reconnectAttempts}
              </Badge>
            )}
          </div>
          <p className="text-xs mt-2">🖱️ Drag to rotate • 🔍 Scroll to zoom • ⚡ Auto-rotating neural matrix</p>
        </div>
      </Html>

      {/* Top-right legends container */}
      <Html fullscreen className="absolute top-4 right-4" zIndexRange={[100, 0]}>
        <div className="flex flex-col space-y-2">
          {connectionHealth.queuedMessages > 0 && (
            <div className="bg-yellow-900/80 text-yellow-200 px-4 py-3 rounded-lg shadow-lg border border-yellow-500/30">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{connectionHealth.queuedMessages} queued messages</span>
              </div>
            </div>
          )}
          {localRetrievalSummaryData && (
            <>
              <div className="bg-black/80 backdrop-blur-sm text-cyan-100 px-4 py-3 rounded-lg shadow-lg border border-cyan-500/30">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-cyan-200">Confidence</span>
                  <Badge variant="outline" className="text-cyan-200 border-cyan-300 bg-transparent">
                    {(localRetrievalSummaryData.confidence_score * 100).toFixed(1)}%
                  </Badge>
                </div>
              </div>
              <div className="bg-black/80 backdrop-blur-sm text-cyan-100 px-4 py-3 rounded-lg shadow-lg border border-cyan-500/30">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-cyan-200">Steps</span>
                  <Badge variant="outline" className="text-cyan-200 border-cyan-300 bg-transparent">
                    {localRetrievalSummaryData.steps_count}
                  </Badge>
                </div>
              </div>
            </>
          )}
        </div>
      </Html>
    </>
  )
}

export function BrainVisualization({ networkId, embeddingDim, onRetrievalSummaryUpdate }: BrainVisualizationProps) {
  const [activity, setActivity] = useState<string>("Neural Network Active")
  const [localRetrievalSummaryData, setLocalRetrievalSummaryData] = useState<RetrievalSummaryData | null>(null)

  // Use the enhanced WebSocket hook
  const { isConnected, lastMessage, connectionHealth, connectionStatus } = useWebSocket(networkId, {
    enabled: true,
    reconnectAttempts: 5,
    reconnectInterval: 1000,
    heartbeatInterval: 30000,
    heartbeatTimeout: 35000,
  })

  // Pass WebSocket updates to the parent component
  useEffect(() => {
    if (lastMessage && lastMessage.event_type === "pattern_retrieved") {
      onRetrievalSummaryUpdate({
        query_text: lastMessage.data.query_text,
        confidence_score: lastMessage.data.confidence,
        steps_count: lastMessage.data.steps?.length || 0,
        retrieved_hash: lastMessage.data.retrieved_hash,
        retrieved_text: lastMessage.data.retrieved_text,
      })
    }
  }, [lastMessage, onRetrievalSummaryUpdate])

  const getConnectionStatusColor = useCallback(() => {
    switch (connectionStatus) {
      case "connected":
        return connectionHealth.connectionHealthy ? "bg-green-400" : "bg-yellow-400"
      case "connecting":
        return "bg-blue-400"
      default:
        return "bg-red-400"
    }
  }, [connectionStatus, connectionHealth.connectionHealthy])

  const getConnectionStatusText = useCallback(() => {
    if (connectionStatus === "connected" && !connectionHealth.connectionHealthy) {
      return "Connection Issues"
    }
    return connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)
  }, [connectionStatus, connectionHealth.connectionHealthy])

  return (
    <div className="relative w-full h-full">
      {" "}
      {/* Changed to w-full h-full */}
      <Canvas
        className="w-full h-full bg-gradient-to-b from-gray-900 to-black rounded-lg overflow-hidden border border-gray-700 shadow-2xl"
        gl={{ antialias: true }}
        camera={{ fov: 75, near: 0.1, far: 1000, position: [0, 5, 18] }}
      >
        <BrainSceneContent
          embeddingDim={embeddingDim}
          lastMessage={lastMessage}
          activity={activity}
          setActivity={setActivity}
          localRetrievalSummaryData={localRetrievalSummaryData}
          setLocalRetrievalSummaryData={setLocalRetrievalSummaryData}
          isConnected={isConnected}
          connectionHealth={connectionHealth}
          connectionStatus={connectionStatus}
          getConnectionStatusColor={getConnectionStatusColor}
          getConnectionStatusText={getConnectionStatusText}
        />
      </Canvas>
    </div>
  )
}
