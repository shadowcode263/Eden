"use client"

import type React from "react"
import { useEffect, useRef, useState, useCallback, useMemo } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, Html } from "@react-three/drei"
import ForceGraph3D, { ForceGraphMethods, NodeObject, LinkObject } from "react-force-graph-3d"
import * as THREE from "three"
import { useWebSocket } from "@/hooks/use-websocket" // Assuming a custom hook for WebSocket
import { Wifi, WifiOff, AlertCircle } from "lucide-react"

// --- TYPE DEFINITIONS for STAG Architecture --- //

interface BrainVisualizationProps {
  networkId: number | null
}

// Represents the structure of the graph data
interface GraphData {
  nodes: NodeObject[]
  links: LinkObject[]
}

// Represents a single node in the graph
interface StagNode extends NodeObject {
  id: number
  error: number
  prototype_sdr?: number[]
}

// Represents a single event from the WebSocket batch
interface GraphEvent {
  event: 'node_added' | 'node_removed' | 'edge_added' | 'edge_removed' | 'node_updated' | 'inference_step' | 'inference_result' | 'narrative_choice'
  payload: any
}

// Represents the batch of events from the WebSocket
interface NetworkUpdateBatch {
  type: 'graph_event_batch'
  events: GraphEvent[]
}

interface SelectedNodeInfo {
  id: number
  error: number
  position: { x: number; y: number; z: number }
}

// --- SCENE & VISUALIZATION COMPONENTS --- //

const SceneSetup = () => {
  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[0, 10, 15]} intensity={1.0} />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        autoRotate
        autoRotateSpeed={0.2}
        minDistance={10}
        maxDistance={100}
        enablePan={false}
      />
    </>
  )
}

const BrainGraphScene = ({ networkId }: BrainVisualizationProps) => {
  const fgRef = useRef<ForceGraphMethods>()
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null)
  const [activity, setActivity] = useState("Awaiting Network Activity...")

  // Fetch the initial state of the graph when the component mounts or networkId changes.
  const fetchInitialGraphState = useCallback(async () => {
    if (!networkId) return
    try {
      const response = await fetch(`http://localhost:8000/api/brain/state/`)
      if (!response.ok) throw new Error("Failed to fetch initial graph state")
      const data = await response.json()
      setGraphData({
        nodes: data.nodes || [],
        links: data.links || [],
      })
      setActivity("Network State Loaded")
    } catch (error) {
      console.error("Error fetching graph state:", error)
      setActivity("Error Loading State")
    }
  }, [networkId])

  useEffect(() => {
    fetchInitialGraphState()
  }, [fetchInitialGraphState])

  // WebSocket handler for real-time updates
  const { lastMessage } = useWebSocket(networkId, { enabled: !!networkId })

  useEffect(() => {
    if (!lastMessage || lastMessage.type !== 'graph_event_batch') return

    const batch: NetworkUpdateBatch = lastMessage
    setGraphData(currentData => {
      let newNodes = [...currentData.nodes]
      let newLinks = [...currentData.links]

      batch.events.forEach(event => {
        const { payload } = event
        switch (event.event) {
          case 'node_added':
            if (!newNodes.some(n => n.id === payload.id)) {
              newNodes.push({ id: payload.id, error: payload.error })
            }
            break
          case 'node_removed':
            newNodes = newNodes.filter(n => n.id !== payload.node_id)
            newLinks = newLinks.filter(l => l.source !== payload.node_id && l.target !== payload.node_id)
            break
          case 'edge_added':
            if (!newLinks.some(l => (l.source === payload.source && l.target === payload.target) || (l.source === payload.target && l.target === payload.source))) {
              newLinks.push({ source: payload.source, target: payload.target })
            }
            break
          case 'edge_removed':
            newLinks = newLinks.filter(l => !((l.source === payload.source && l.target === payload.target) || (l.source === payload.target && l.target === payload.source)))
            break
          case 'node_updated':
            const nodeToUpdate = newNodes.find(n => n.id === payload.id)
            if (nodeToUpdate) {
              ;(nodeToUpdate as StagNode).error = payload.error
            }
            break
          case 'inference_step':
          case 'narrative_choice':
            const nodeId = event.event === 'inference_step' ? payload.processed_node_id : payload.action
            const node = newNodes.find(n => n.id === nodeId)
            if (node && fgRef.current) {
                fgRef.current.emitParticle(node)
            }
            setActivity(`Processing Concept: ${nodeId}`)
            break
          case 'inference_result':
            setActivity(`Prediction: ${payload.predicted_next_node_id}`)
            break
        }
      })
      return { nodes: newNodes, links: newLinks }
    })
  }, [lastMessage])

  const handleNodeClick = useCallback((node: NodeObject) => {
    const stagNode = node as StagNode
    if (stagNode && stagNode.x !== undefined && stagNode.y !== undefined && stagNode.z !== undefined) {
      setSelectedNode({
        id: stagNode.id,
        error: stagNode.error,
        position: { x: stagNode.x, y: stagNode.y, z: stagNode.z },
      })
      // Center camera on the clicked node
      fgRef.current?.cameraPosition({ x: stagNode.x, y: stagNode.y, z: stagNode.z + 20 }, stagNode, 1000)
    }
  }, [])

  return (
    <div className="relative w-full h-full">
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="id"
        nodeColor={node => {
          const error = (node as StagNode).error || 0
          return error > 0.5 ? '#ff4d4d' : '#4d94ff'
        }}
        nodeVal={node => {
            const error = (node as StagNode).error || 0
            return 0.5 + error * 2
        }}
        linkColor={() => 'rgba(255,255,255,0.2)'}
        linkWidth={0.5}
        onNodeClick={handleNodeClick}
        particleColor={() => '#00ff88'}
        particleWidth={4}
      />
      {selectedNode && (
        <Html
          position={new THREE.Vector3(selectedNode.position.x, selectedNode.position.y, selectedNode.position.z)}
          center
          distanceFactor={15}
          style={{ pointerEvents: "none" }}
        >
          <div
            className="bg-slate-900/90 text-blue-300 p-3 rounded-lg shadow-2xl text-sm whitespace-nowrap border border-blue-400/40 backdrop-blur-sm"
          >
            <p className="font-mono font-bold">Concept Node: {selectedNode.id}</p>
            <p className="font-mono text-xs text-blue-400">
              Error Level: {selectedNode.error.toFixed(4)}
            </p>
          </div>
        </Html>
      )}
       <div className="absolute top-4 left-4 text-blue-300 bg-slate-900/80 backdrop-blur-md px-4 py-2 rounded-lg border border-blue-400/40">
          <p className="text-sm font-medium font-mono">{activity}</p>
        </div>
    </div>
  )
}

// --- EXPORTED ROOT COMPONENT --- //
export function BrainVisualization({ networkId }: BrainVisualizationProps) {
  const { isConnected, connectionStatus } = useWebSocket(networkId, { enabled: !!networkId })

  return (
    <div className="relative w-full h-full">
      <Canvas
        className="w-full h-full rounded-lg"
        gl={{ antialias: true, alpha: false }}
        camera={{ position: [0, 0, 80], fov: 50 }}
      >
        <SceneSetup />
        {networkId && <BrainGraphScene networkId={networkId} />}
      </Canvas>

      <div className="absolute top-4 right-4 flex items-center space-x-2 text-blue-300 bg-slate-900/80 backdrop-blur-md px-4 py-2 rounded-lg border border-blue-400/40">
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-400" />
          )}
          <span className="text-xs font-mono">{connectionStatus}</span>
      </div>

      {!networkId && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm rounded-lg">
          <div className="text-center text-blue-300">
            <AlertCircle className="w-16 h-16 mx-auto mb-6 text-blue-400" />
            <p className="text-xl font-medium font-mono mb-2">No Network Selected</p>
            <p className="text-sm text-blue-400 font-mono">Select a network to begin visualization</p>
          </div>
        </div>
      )}
    </div>
  )
}
