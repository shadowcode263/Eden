"use client"

import React from "react"
import { useEffect, useRef, useState, useCallback, useMemo } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Wifi, WifiOff, AlertCircle, Box, Layers, RefreshCw } from "lucide-react"
import { useWebSocket } from "@/hooks/use-websocket"
import dynamic from "next/dynamic"

// Lazy load 3D components for better performance
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
  loading: () => <LoadingBrain />,
})

// === BRAIN-LIKE ARCHITECTURE ===
// Sensory Layer: Input processing and data structures
interface NeuralNode {
  id: string
  x: number
  y: number
  z?: number
  color: string
  active: boolean
  error: number
  type: string
  lastActive: number
}

interface NeuralLink {
  source: string
  target: string
  strength: number
  color: string
  age: number
}

interface BrainState {
  nodes: NeuralNode[]
  links: NeuralLink[]
  activity: string
  stats: {
    total: number
    active: number
    errors: number
  }
}

interface VisualizationProps {
  networkId: number | null
}

// Memory Layer: Optimized state management
class BrainMemory {
  private nodeCache = new Map<string, NeuralNode>()
  private linkCache = new Map<string, NeuralLink>()
  private positionCache = new Map<string, { x: number; y: number; z?: number }>()

  updateNode(node: any): NeuralNode {
    const id = node.id.toString()
    const cached = this.nodeCache.get(id)
    const position = this.positionCache.get(id) || this.generatePosition(id)

    const neuralNode: NeuralNode = {
      id,
      x: position.x,
      y: position.y,
      z: position.z,
      color: this.getNodeColor(node),
      active: node.last_active_iter > 0,
      error: node.error || 0,
      type: node.type || "sensory",
      lastActive: node.last_active_iter || 0,
    }

    this.nodeCache.set(id, neuralNode)
    this.positionCache.set(id, { x: neuralNode.x, y: neuralNode.y, z: neuralNode.z })
    return neuralNode
  }

  updateLink(link: any): NeuralLink {
    const key = `${link.source}-${link.target}`
    const strength = Math.max(0.1, 1 - (link.age || 0) / 50)

    const neuralLink: NeuralLink = {
      source: link.source.toString(),
      target: link.target.toString(),
      strength,
      color: `rgba(100, 116, 139, ${Math.max(0.2, strength)})`,
      age: link.age || 0,
    }

    this.linkCache.set(key, neuralLink)
    return neuralLink
  }

  private generatePosition(id: string): { x: number; y: number; z?: number } {
    const hash = this.hashCode(id)
    const angle = (hash % 360) * (Math.PI / 180)
    const radius = 80 + (hash % 100)

    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      z: (hash % 200) - 100,
    }
  }

  private getNodeColor(node: any): string {
    const error = node.error || 0
    const isActive = node.last_active_iter > 0

    if (isActive) return "#00ff88"
    return error > 0.5 ? "#ff4d4d" : "#4d94ff"
  }

  private hashCode(str: string): number {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i)
      hash = (hash << 5) - hash + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash)
  }

  clear() {
    this.nodeCache.clear()
    this.linkCache.clear()
    this.positionCache.clear()
  }
}

// Processing Layer: Data transformation and optimization
class BrainProcessor {
  private memory = new BrainMemory()

  processGraphData(rawData: any): BrainState {
    const nodes = (rawData.nodes || []).map((node: any) => this.memory.updateNode(node))
    const links = (rawData.links || []).map((link: any) => this.memory.updateLink(link))

    const stats = {
      total: nodes.length,
      active: nodes.filter((n) => n.active).length,
      errors: nodes.filter((n) => n.error > 0.3).length,
    }

    return {
      nodes,
      links,
      activity: `Network Active - ${stats.total} nodes`,
      stats,
    }
  }

  processWebSocketMessage(message: any, currentState: BrainState): BrainState {
    if (message.type === "graph_state_update") {
      return this.processGraphData(message.payload)
    }

    if (message.type === "game_event") {
      const activity = this.processGameEvent(message.payload)
      return { ...currentState, activity }
    }

    return currentState
  }

  private processGameEvent(payload: any): string {
    switch (payload.event_type) {
      case "action":
        return `Action: ${payload.data.action} (Step ${payload.data.step})`
      case "reward":
        return `Reward: ${payload.data.reward.toFixed(2)}`
      case "episode_start":
        return `Episode ${payload.data.episode} Started`
      case "episode_end":
        return `Episode Complete - Reward: ${payload.data.total_reward.toFixed(2)}`
      case "training_complete":
        return `Training Complete - ${payload.data.total_episodes} episodes`
      default:
        return "Processing..."
    }
  }

  reset() {
    this.memory.clear()
  }
}

// Motor Layer: Optimized rendering components
const LoadingBrain = () => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center text-blue-300">
      <div className="relative mb-6">
        <div className="w-16 h-16 mx-auto border-4 border-blue-400 rounded-full animate-pulse"></div>
        <div className="absolute inset-0 w-16 h-16 mx-auto border-t-4 border-green-400 rounded-full animate-spin"></div>
      </div>
      <p className="text-xl font-medium font-mono mb-2">NEURAL INTERFACE LOADING</p>
      <p className="text-sm text-blue-400 font-mono">Initializing pathways...</p>
    </div>
  </div>
)

const EmptyBrainState = () => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center text-blue-300">
      <div className="relative mb-6">
        <div className="w-20 h-20 mx-auto border-2 border-blue-400 rounded-full opacity-50"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-12 h-12 border-2 border-green-400 rounded-full opacity-30"></div>
        </div>
      </div>
      <p className="text-xl font-medium font-mono mb-2">NETWORK OFFLINE</p>
      <p className="text-sm text-blue-400 font-mono">Initialize neural pathways to begin</p>
    </div>
  </div>
)

// Optimized 2D Canvas Renderer
const Canvas2D = React.memo(
  ({
    brainState,
    selectedNode,
    onNodeClick,
  }: {
    brainState: BrainState
    selectedNode: string | null
    onNodeClick: (nodeId: string | null) => void
  }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const animationRef = useRef<number>()

    const render = useCallback(() => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      const { width, height } = canvas.getBoundingClientRect()
      canvas.width = width * devicePixelRatio
      canvas.height = height * devicePixelRatio
      ctx.scale(devicePixelRatio, devicePixelRatio)

      ctx.clearRect(0, 0, width, height)

      if (brainState.nodes.length === 0) {
        return
      }

      const centerX = width / 2
      const centerY = height / 2

      // Calculate center of mass
      const totalX = brainState.nodes.reduce((sum, node) => sum + node.x, 0)
      const totalY = brainState.nodes.reduce((sum, node) => sum + node.y, 0)
      const avgX = totalX / brainState.nodes.length
      const avgY = totalY / brainState.nodes.length
      const offsetX = centerX - avgX
      const offsetY = centerY - avgY

      // Render links
      ctx.globalAlpha = 0.6
      brainState.links.forEach((link) => {
        const source = brainState.nodes.find((n) => n.id === link.source)
        const target = brainState.nodes.find((n) => n.id === link.target)

        if (source && target) {
          ctx.strokeStyle = link.color
          ctx.lineWidth = Math.max(0.5, link.strength * 2)
          ctx.beginPath()
          ctx.moveTo(source.x + offsetX, source.y + offsetY)
          ctx.lineTo(target.x + offsetX, target.y + offsetY)
          ctx.stroke()
        }
      })

      // Render nodes
      ctx.globalAlpha = 1
      brainState.nodes.forEach((node) => {
        const x = node.x + offsetX
        const y = node.y + offsetY
        const size = 8

        // Node glow for active nodes
        if (node.active) {
          ctx.shadowColor = node.color
          ctx.shadowBlur = 15
        }

        // Main node
        ctx.fillStyle = node.color
        ctx.beginPath()
        ctx.arc(x, y, size, 0, Math.PI * 2)
        ctx.fill()

        // Selection highlight
        if (selectedNode === node.id) {
          ctx.strokeStyle = "#fbbf24"
          ctx.lineWidth = 3
          ctx.beginPath()
          ctx.arc(x, y, size + 5, 0, Math.PI * 2)
          ctx.stroke()
        }

        // Node border
        ctx.strokeStyle = node.active ? "#00ff88" : "#1e293b"
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.arc(x, y, size, 0, Math.PI * 2)
        ctx.stroke()

        ctx.shadowBlur = 0
      })

      animationRef.current = requestAnimationFrame(render)
    }, [brainState, selectedNode])

    useEffect(() => {
      render()
      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }, [render])

    const handleClick = useCallback(
      (event: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = event.currentTarget
        const rect = canvas.getBoundingClientRect()
        const x = event.clientX - rect.left
        const y = event.clientY - rect.top

        const centerX = rect.width / 2
        const centerY = rect.height / 2
        const totalX = brainState.nodes.reduce((sum, node) => sum + node.x, 0)
        const totalY = brainState.nodes.reduce((sum, node) => sum + node.y, 0)
        const avgX = totalX / brainState.nodes.length
        const avgY = totalY / brainState.nodes.length
        const offsetX = centerX - avgX
        const offsetY = centerY - avgY

        const clickedNode = brainState.nodes.find((node) => {
          const nodeX = node.x + offsetX
          const nodeY = node.y + offsetY
          const distance = Math.sqrt((nodeX - x) ** 2 + (nodeY - y) ** 2)
          return distance <= 13 // 8 (size) + 5 (tolerance)
        })

        onNodeClick(clickedNode?.id || null)
      },
      [brainState.nodes, onNodeClick],
    )

    return (
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        onClick={handleClick}
        style={{ background: "transparent" }}
      />
    )
  },
)

Canvas2D.displayName = "Canvas2D"

// Optimized 3D Renderer
const Canvas3D = React.memo(
  ({
    brainState,
    selectedNode,
    onNodeClick,
  }: {
    brainState: BrainState
    selectedNode: string | null
    onNodeClick: (nodeId: string | null) => void
  }) => {
    const fgRef = useRef<any>()

    const graph3DData = useMemo(
      () => ({
        nodes: brainState.nodes.map((node) => ({
          id: node.id,
          color: node.color,
          size: node.active ? 12 : 8,
          active: node.active,
          error: node.error,
          type: node.type,
        })),
        links: brainState.links.map((link) => ({
          source: link.source,
          target: link.target,
          color: link.color,
          width: Math.max(0.5, link.strength * 3),
          opacity: link.strength,
        })),
      }),
      [brainState],
    )

    const handleNodeClick = useCallback(
      (node: any) => {
        onNodeClick(node.id)
        if (fgRef.current && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
          fgRef.current.cameraPosition({ x: node.x, y: node.y, z: node.z + 150 }, node, 1500)
        }
      },
      [onNodeClick],
    )

    if (brainState.nodes.length === 0) {
      return <EmptyBrainState />
    }

    return (
      <ForceGraph3D
        ref={fgRef}
        graphData={graph3DData}
        nodeLabel="id"
        nodeColor={(node: any) => node.color}
        nodeVal={(node: any) => node.size}
        nodeOpacity={0.9}
        linkColor={(link: any) => link.color}
        linkWidth={(link: any) => link.width}
        linkOpacity={(link: any) => link.opacity}
        onNodeClick={handleNodeClick}
        backgroundColor="rgba(15, 23, 42, 1)"
        showNavInfo={false}
        controlType="orbit"
        enableNodeDrag={false}
        d3AlphaDecay={0.01}
        d3VelocityDecay={0.2}
        warmupTicks={100}
        cooldownTicks={200}
      />
    )
  },
)

Canvas3D.displayName = "Canvas3D"

// Main Brain Visualization Component
export function BrainVisualization({ networkId }: VisualizationProps) {
  const [brainState, setBrainState] = useState<BrainState>({
    nodes: [],
    links: [],
    activity: "Awaiting Network Activity...",
    stats: { total: 0, active: 0, errors: 0 },
  })
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [is3D, setIs3D] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const processor = useMemo(() => new BrainProcessor(), [])
  const { isConnected, lastMessage } = useWebSocket(networkId, {
    url: "ws://localhost:8000",
    enabled: networkId !== null,
  })

  // Fetch initial state
  const fetchBrainState = useCallback(async () => {
    if (!networkId) return

    setIsRefreshing(true)
    try {
      const response = await fetch(`http://localhost:8000/api/brain/state/`)
      if (!response.ok) throw new Error("Failed to fetch brain state")

      const data = await response.json()
      const newState = processor.processGraphData(data)
      setBrainState(newState)
    } catch (error) {
      console.error("Error fetching brain state:", error)
      setBrainState((prev) => ({ ...prev, activity: "Error Loading State" }))
    } finally {
      setIsRefreshing(false)
    }
  }, [networkId, processor])

  // WebSocket message processing
  useEffect(() => {
    if (!lastMessage) return

    setBrainState((currentState) => processor.processWebSocketMessage(lastMessage, currentState))
  }, [lastMessage, processor])

  // Initial load and auto-refresh
  useEffect(() => {
    fetchBrainState()

    const interval = setInterval(fetchBrainState, 5000)
    return () => clearInterval(interval)
  }, [fetchBrainState])

  // Cleanup on unmount
  useEffect(() => {
    return () => processor.reset()
  }, [processor])

  const selectedNodeData = useMemo(
    () => brainState.nodes.find((n) => n.id === selectedNode),
    [brainState.nodes, selectedNode],
  )

  if (!networkId) {
    return (
      <Card className="w-full h-full bg-slate-900 border-slate-800">
        <CardContent className="p-0 h-full flex items-center justify-center">
          <div className="text-center text-blue-300">
            <AlertCircle className="w-16 h-16 mx-auto mb-6 text-blue-400" />
            <p className="text-xl font-medium font-mono mb-2">NO NETWORK SELECTED</p>
            <p className="text-sm text-blue-400 font-mono">Initialize neural interface to begin</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full h-full bg-slate-900 border-slate-800 overflow-hidden">
      <CardContent className="p-0 h-full relative">
        {/* Control Panel */}
        <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between">
          <div className="text-blue-300 bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-lg border border-blue-400/40">
            <p className="text-sm font-medium font-mono">{brainState.activity}</p>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              onClick={() => setIs3D(!is3D)}
              variant="outline"
              size="sm"
              className="bg-slate-800/90 border-blue-400/40 text-blue-300 hover:bg-slate-700/90"
            >
              {is3D ? <Layers className="w-4 h-4 mr-2" /> : <Box className="w-4 h-4 mr-2" />}
              {is3D ? "2D" : "3D"}
            </Button>

            <Button
              onClick={fetchBrainState}
              variant="outline"
              size="sm"
              disabled={isRefreshing}
              className="bg-slate-800/90 border-blue-400/40 text-blue-300 hover:bg-slate-700/90"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>

            <div className="flex items-center space-x-2 text-blue-300 bg-slate-900/90 backdrop-blur-md px-3 py-2 rounded-lg border border-blue-400/40">
              {isConnected ? <Wifi className="w-4 h-4 text-green-400" /> : <WifiOff className="w-4 h-4 text-red-400" />}
              <span className="text-xs font-mono">{isConnected ? "Live" : "Offline"}</span>
            </div>
          </div>
        </div>

        {/* Visualization */}
        {brainState.nodes.length === 0 ? (
          <EmptyBrainState />
        ) : is3D ? (
          <Canvas3D brainState={brainState} selectedNode={selectedNode} onNodeClick={setSelectedNode} />
        ) : (
          <Canvas2D brainState={brainState} selectedNode={selectedNode} onNodeClick={setSelectedNode} />
        )}

        {/* Info Panel */}
        <div className="absolute bottom-4 left-4 right-4 z-10 flex items-end justify-between">
          {selectedNodeData && (
            <div className="bg-slate-900/95 text-blue-300 p-3 rounded-lg border border-blue-400/40 backdrop-blur-sm">
              <p className="font-mono font-bold">Node: {selectedNodeData.id}</p>
              <p className="font-mono text-xs text-blue-400">Error: {selectedNodeData.error.toFixed(4)}</p>
              <p className="font-mono text-xs text-green-400">Type: {selectedNodeData.type}</p>
              <p className="font-mono text-xs text-yellow-400">Last Active: {selectedNodeData.lastActive}</p>
            </div>
          )}

          <div className="flex items-center space-x-2">
            <Badge
              variant="outline"
              className="bg-slate-800/90 text-slate-200 border-slate-700 font-mono backdrop-blur-md"
            >
              {brainState.stats.total} nodes • {brainState.links.length} edges
            </Badge>
            <Badge
              variant="outline"
              className="bg-green-500/20 text-green-400 border-green-500/30 font-mono backdrop-blur-md"
            >
              {brainState.stats.active} active
            </Badge>
            <Badge
              variant="outline"
              className="bg-red-500/20 text-red-400 border-red-500/30 font-mono backdrop-blur-md"
            >
              {brainState.stats.errors} errors
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
