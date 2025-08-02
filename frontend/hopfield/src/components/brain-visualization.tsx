"use client"

import type React from "react"
import { useEffect, useRef, useState, useCallback } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Wifi, WifiOff, AlertCircle, Box, Layers, RefreshCw } from "lucide-react"
import { useWebSocket } from "@/hooks/use-websocket"
import dynamic from "next/dynamic"

// Dynamically import ForceGraph3D to avoid SSR issues
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <div className="text-center text-blue-300">
        <div className="w-16 h-16 mx-auto border-4 border-blue-400 rounded-full animate-pulse mb-4"></div>
        <p className="text-sm font-mono">Loading 3D Engine...</p>
      </div>
    </div>
  ),
})

// Dynamically import THREE to avoid SSR issues
const THREE = dynamic(() => import("three"), { ssr: false })

interface Node {
  id: string
  x: number
  y: number
  z?: number
  size: number
  color: string
  active?: boolean
  error?: number
  last_active_iter?: number
  type?: string
}

interface Link {
  source: string
  target: string
  strength: number
  color: string
  age?: number
}

interface GraphData {
  nodes: Node[]
  links: Link[]
}

interface BrainVisualizationProps {
  networkId: number | null
}

interface SelectedNodeInfo {
  id: string
  error: number
  position: { x: number; y: number; z?: number }
  type?: string
  last_active?: number
}

// 2D Visualization Component (Always Centered)
function BrainVisualization2D({
  networkId,
  graphData,
  selectedNode,
  setSelectedNode,
  activity,
  isConnected,
  handleCanvasClick,
}: {
  networkId: number | null
  graphData: GraphData
  selectedNode: SelectedNodeInfo | null
  setSelectedNode: (node: SelectedNodeInfo | null) => void
  activity: string
  isConnected: boolean
  handleCanvasClick: (event: React.MouseEvent<HTMLCanvasElement>) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Canvas rendering with always-centered layout
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      canvas.style.width = rect.width + "px"
      canvas.style.height = rect.height + "px"
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const centerX = canvas.clientWidth / 2
      const centerY = canvas.clientHeight / 2

      if (graphData.nodes.length === 0) {
        // Show empty state with sci-fi styling
        ctx.fillStyle = "#64748b"
        ctx.font = "18px 'JetBrains Mono', monospace"
        ctx.textAlign = "center"
        ctx.fillText("NETWORK OFFLINE", centerX, centerY - 20)

        ctx.font = "14px 'JetBrains Mono', monospace"
        ctx.fillStyle = "#94a3b8"
        ctx.fillText("Initialize neural pathways to begin", centerX, centerY + 10)

        // Draw pulsing brain icon
        const time = Date.now() * 0.003
        const pulse = 0.8 + 0.2 * Math.sin(time)
        const brainSize = 50 * pulse

        ctx.strokeStyle = `rgba(100, 116, 139, ${pulse})`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY - 80, brainSize / 2, 0, Math.PI * 2)
        ctx.stroke()

        // Add scanning lines effect
        ctx.strokeStyle = `rgba(59, 130, 246, 0.3)`
        ctx.lineWidth = 1
        for (let i = 0; i < 3; i++) {
          const offset = (time * 50 + i * 30) % 200
          ctx.beginPath()
          ctx.moveTo(centerX - 100, centerY - 80 + offset - 100)
          ctx.lineTo(centerX + 100, centerY - 80 + offset - 100)
          ctx.stroke()
        }
      } else {
        // Calculate center of mass for existing nodes
        let totalX = 0
        let totalY = 0
        graphData.nodes.forEach((node) => {
          totalX += node.x
          totalY += node.y
        })
        const avgX = totalX / graphData.nodes.length
        const avgY = totalY / graphData.nodes.length

        // Calculate offset to center the graph
        const offsetX = centerX - avgX
        const offsetY = centerY - avgY

        // Draw links first (so they appear behind nodes) with centering offset
        graphData.links.forEach((link) => {
          const sourceNode = graphData.nodes.find((n) => n.id === link.source)
          const targetNode = graphData.nodes.find((n) => n.id === link.target)

          if (sourceNode && targetNode) {
            // Add glow effect for strong connections
            if (link.strength > 0.7) {
              ctx.shadowColor = link.color
              ctx.shadowBlur = 10
            }

            ctx.strokeStyle = link.color || "#64748b"
            ctx.lineWidth = Math.max(0.5, link.strength * 3)
            ctx.globalAlpha = 0.8
            ctx.beginPath()
            ctx.moveTo(sourceNode.x + offsetX, sourceNode.y + offsetY)
            ctx.lineTo(targetNode.x + offsetX, targetNode.y + offsetY)
            ctx.stroke()

            ctx.shadowBlur = 0
            ctx.globalAlpha = 1
          }
        })

        // Draw nodes with enhanced styling and centering offset
        graphData.nodes.forEach((node) => {
          const nodeX = node.x + offsetX
          const nodeY = node.y + offsetY

          // Node glow effect for active nodes
          if (node.active) {
            ctx.shadowColor = node.color
            ctx.shadowBlur = 20
          }

          // Main node body - use constant size
          const nodeSize = 8 // Constant size for all nodes
          ctx.fillStyle = node.color || "#3b82f6"
          ctx.beginPath()
          ctx.arc(nodeX, nodeY, nodeSize, 0, Math.PI * 2)
          ctx.fill()

          // Active node pulse ring
          if (node.active) {
            const time = Date.now() * 0.005
            const pulseRadius = nodeSize + 5 + 3 * Math.sin(time)
            ctx.strokeStyle = `rgba(0, 255, 136, ${0.5 + 0.3 * Math.sin(time)})`
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.arc(nodeX, nodeY, pulseRadius, 0, Math.PI * 2)
            ctx.stroke()
          }

          // Node border
          ctx.strokeStyle = node.active ? "#00ff88" : "#1e293b"
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(nodeX, nodeY, nodeSize, 0, Math.PI * 2)
          ctx.stroke()

          // Node ID with monospace font
          ctx.fillStyle = "#ffffff"
          ctx.font = "10px 'JetBrains Mono', monospace"
          ctx.textAlign = "center"
          ctx.fillText(node.id, nodeX, nodeY + 3)

          // Error indicator
          if (node.error && node.error > 0.3) {
            ctx.fillStyle = "#ef4444"
            ctx.beginPath()
            ctx.arc(nodeX + nodeSize - 2, nodeY - nodeSize + 2, 3, 0, Math.PI * 2)
            ctx.fill()
          }

          ctx.shadowBlur = 0

          // Update node position for click detection (store centered positions)
          node.x = nodeX
          node.y = nodeY
        })

        // Draw selection highlight
        if (selectedNode) {
          const node = graphData.nodes.find((n) => n.id === selectedNode.id)
          if (node) {
            const nodeSize = 8 // Use same constant size
            ctx.strokeStyle = "#fbbf24"
            ctx.lineWidth = 3
            ctx.beginPath()
            ctx.arc(node.x, node.y, nodeSize + 8, 0, Math.PI * 2)
            ctx.stroke()
          }
        }
      }

      requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
    }
  }, [graphData, selectedNode])

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full cursor-pointer"
      onClick={handleCanvasClick}
      style={{ background: "transparent" }}
    />
  )
}

// 3D Visualization Component using react-force-graph-3d (Smooth Animation)
function BrainVisualization3D({
  networkId,
  graphData,
  selectedNode,
  setSelectedNode,
  activity,
  isConnected,
}: {
  networkId: number | null
  graphData: GraphData
  selectedNode: SelectedNodeInfo | null
  setSelectedNode: (node: SelectedNodeInfo | null) => void
  activity: string
  isConnected: boolean
}) {
  const fgRef = useRef<any>()
  const [threeLoaded, setThreeLoaded] = useState(false)

  useEffect(() => {
    // Dynamically load THREE.js
    import("three").then(() => {
      setThreeLoaded(true)
    })
  }, [])

  // Transform data for react-force-graph-3d with centering
  const graph3DData = {
    nodes: graphData.nodes.map((node) => ({
      id: node.id,
      color: node.color,
      size: node.size || 6,
      active: node.active,
      error: node.error || 0,
      type: node.type,
      last_active_iter: node.last_active_iter || 0,
    })),
    links: graphData.links.map((link) => ({
      source: link.source,
      target: link.target,
      color: link.color,
      width: Math.max(0.5, link.strength * 3),
      opacity: link.strength,
    })),
  }

  const handleNodeClick = useCallback(
    (node: any) => {
      if (node && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
        setSelectedNode({
          id: node.id,
          error: node.error || 0,
          type: node.type,
          last_active: node.last_active_iter,
          position: { x: node.x, y: node.y, z: node.z },
        })
        // Smooth camera transition to the clicked node
        fgRef.current?.cameraPosition({ x: node.x, y: node.y, z: node.z + 150 }, node, 2000)
      }
    },
    [setSelectedNode],
  )

  if (graphData.nodes.length === 0 || !threeLoaded) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-blue-300">
          <div className="relative mb-6">
            <div className="w-16 h-16 mx-auto border-4 border-blue-400 rounded-full animate-pulse"></div>
            <div className="absolute inset-0 w-16 h-16 mx-auto border-t-4 border-green-400 rounded-full animate-spin"></div>
          </div>
          <p className="text-xl font-medium font-mono mb-2">3D NETWORK INITIALIZING</p>
          <p className="text-sm text-blue-400 font-mono">Preparing neural pathways...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full flex items-center justify-center">
      <ForceGraph3D
        ref={fgRef}
        graphData={graph3DData}
        width={undefined} // Let it auto-size
        height={undefined} // Let it auto-size
        nodeLabel="id"
        nodeColor={(node: any) => node.color}
        nodeVal={(node: any) => {
          const isActive = node.active
          return isActive ? 12 : 8 // Constant sizes, just different for active vs inactive
        }}
        nodeOpacity={0.9}
        linkColor={(link: any) => link.color}
        linkWidth={(link: any) => link.width}
        linkOpacity={(link: any) => link.opacity}
        onNodeClick={handleNodeClick}
        backgroundColor="rgba(15, 23, 42, 1)"
        showNavInfo={false}
        controlType="orbit"
        enableNodeDrag={false}
        enableNavigationControls={true}
        // Smooth animation parameters
        d3AlphaDecay={0.005} // Slower decay for smoother animation (was 0.01)
        d3VelocityDecay={0.15} // Lower velocity decay for smoother movement (was 0.3)
        d3AlphaMin={0.001} // Lower minimum alpha for longer simulation
        warmupTicks={200} // More warmup ticks for better initial positioning (was 100)
        cooldownTicks={500} // More cooldown ticks for smoother settling (was 200)
        // Additional smoothing parameters
        d3ReheatSimulation={false} // Prevent jarring reheat
        nodeRelSize={4} // Relative node size
        linkDirectionalParticles={0} // No particles for smoother performance
        linkDirectionalParticleSpeed={0.006} // Slower particles if enabled
        // Camera controls for smoother interaction
        cameraPosition={{ x: 0, y: 0, z: 400 }} // Better initial camera position
      />
    </div>
  )
}

export function BrainVisualization({ networkId }: BrainVisualizationProps) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null)
  const [activity, setActivity] = useState("Awaiting Network Activity...")
  const [is3D, setIs3D] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const { isConnected, lastMessage } = useWebSocket(networkId, {
    url: "ws://localhost:8000",
    enabled: networkId !== null,
  })

  // Helper function to determine node color based on state
  const getNodeColor = useCallback((node: any) => {
    const error = node.error || 0
    const isActive = node.last_active_iter && node.last_active_iter > 0

    if (isActive) return "#00ff88" // Active nodes are bright green
    return error > 0.5 ? "#ff4d4d" : "#4d94ff" // High error = red, low error = blue
  }, [])

  // Fetch initial graph state
  const fetchInitialGraphState = useCallback(async () => {
    if (!networkId) return
    setIsRefreshing(true)
    try {
      const response = await fetch(`http://localhost:8000/api/brain/state/`)
      if (!response.ok) throw new Error("Failed to fetch initial graph state")
      const data = await response.json()

      // Transform the data to match our interface - positions will be centered in rendering
      const nodeCount = data.nodes?.length || 0
      const transformedNodes = (data.nodes || []).map((node: any, index: number) => {
        // Use more natural force-directed positioning - centering handled in render
        const nodeCount = data.nodes?.length || 0
        let x, y

        if (nodeCount <= 1) {
          // Single node at origin (will be centered in render)
          x = 0
          y = 0
        } else {
          // Use a more organic layout with some randomness
          const angle = (index * 2 * Math.PI) / nodeCount + (Math.random() - 0.5) * 0.5
          const radius = 80 + Math.random() * 100 // Variable radius for more natural look
          x = Math.cos(angle) * radius + (Math.random() - 0.5) * 40
          y = Math.sin(angle) * radius + (Math.random() - 0.5) * 40
        }

        return {
          id: node.id.toString(),
          x,
          y,
          size: 8, // Constant size for all nodes
          color: getNodeColor(node),
          active: node.last_active_iter > 0,
          error: node.error || 0,
          last_active_iter: node.last_active_iter || 0,
          type: node.type || "sensory",
        }
      })

      const transformedLinks = (data.links || []).map((link: any) => ({
        source: link.source.toString(),
        target: link.target.toString(),
        strength: Math.max(0.1, 1 - (link.age || 0) / 50),
        color: `rgba(100, 116, 139, ${Math.max(0.2, 1 - (link.age || 0) / 50)})`,
        age: link.age || 0,
      }))

      setGraphData({
        nodes: transformedNodes,
        links: transformedLinks,
      })
      setActivity(`Network State Loaded - ${nodeCount} nodes`)
    } catch (error) {
      console.error("Error fetching graph state:", error)
      setActivity("Error Loading State")
    } finally {
      setIsRefreshing(false)
    }
  }, [networkId, getNodeColor])

  useEffect(() => {
    fetchInitialGraphState()
  }, [fetchInitialGraphState])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!networkId) return

    const interval = setInterval(() => {
      fetchInitialGraphState()
    }, 5000) // 5 seconds

    return () => clearInterval(interval)
  }, [networkId, fetchInitialGraphState])

  // Handle WebSocket messages - FIXED: Removed graphData.nodes from dependencies
  useEffect(() => {
    if (!lastMessage) return

    if (lastMessage.type === "graph_state_update") {
      const payload = lastMessage.payload
      if (payload) {
        const nodeCount = payload.nodes?.length || 0
        const transformedNodes = (payload.nodes || []).map((node: any, index: number) => {
          // Maintain existing positions if node exists, otherwise create new natural position
          const existingNode = graphData.nodes.find((n) => n.id === node.id.toString())
          let x, y

          if (existingNode) {
            // Keep existing position (relative to center)
            x = existingNode.x
            y = existingNode.y
          } else {
            // New node - calculate more natural position
            const nodeCount = payload.nodes?.length || 0
            const angle = (index * 2 * Math.PI) / nodeCount + (Math.random() - 0.5) * 0.5
            const radius = 80 + Math.random() * 100
            x = Math.cos(angle) * radius + (Math.random() - 0.5) * 40
            y = Math.sin(angle) * radius + (Math.random() - 0.5) * 40
          }

          return {
            id: node.id.toString(),
            x,
            y,
            size: 8, // Constant size for all nodes
            color: getNodeColor(node),
            active: node.last_active_iter > 0,
            error: node.error || 0,
            last_active_iter: node.last_active_iter || 0,
            type: node.type || "sensory",
          }
        })

        const transformedLinks = (payload.links || []).map((link: any) => ({
          source: link.source.toString(),
          target: link.target.toString(),
          strength: Math.max(0.1, 1 - (link.age || 0) / 50),
          color: `rgba(100, 116, 139, ${Math.max(0.2, 1 - (link.age || 0) / 50)})`,
          age: link.age || 0,
        }))

        setGraphData({
          nodes: transformedNodes,
          links: transformedLinks,
        })
        setActivity(`Graph Updated - ${nodeCount} nodes`)
      }
    }
  }, [lastMessage, getNodeColor])

  // Handle canvas click for node selection (with centered coordinates)
  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = event.currentTarget
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top

      // Find clicked node (nodes are already positioned with centering offset)
      const clickedNode = graphData.nodes.find((node) => {
        const distance = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
        return distance <= 8 + 5 // Use constant size (8) + click tolerance (5)
      })

      if (clickedNode) {
        setSelectedNode({
          id: clickedNode.id,
          error: clickedNode.error || 0,
          type: clickedNode.type,
          last_active: clickedNode.last_active_iter,
          position: { x: clickedNode.x, y: clickedNode.y },
        })
      } else {
        setSelectedNode(null)
      }
    },
    [graphData.nodes],
  )

  return (
    <Card className="w-full h-full bg-slate-900 border-slate-800 overflow-hidden shadow-2xl">
      <CardContent className="p-0 h-full relative">
        {/* Integrated Controls Panel - Top */}
        <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between">
          {/* Left Side - Activity Status */}
          <div className="text-blue-300 bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-lg border border-blue-400/40 shadow-lg">
            <p className="text-sm font-medium font-mono">{activity}</p>
          </div>

          {/* Right Side - Controls */}
          <div className="flex items-center space-x-2">
            {/* 2D/3D Toggle */}
            <Button
              onClick={() => setIs3D(!is3D)}
              variant="outline"
              size="sm"
              className="bg-slate-800/90 border-blue-400/40 text-blue-300 hover:bg-slate-700/90 hover:border-blue-300/60 transition-all duration-200 backdrop-blur-md"
            >
              {is3D ? (
                <>
                  <Layers className="w-4 h-4 mr-2" />
                  2D
                </>
              ) : (
                <>
                  <Box className="w-4 h-4 mr-2" />
                  3D
                </>
              )}
            </Button>

            {/* Manual Refresh */}
            <Button
              onClick={fetchInitialGraphState}
              variant="outline"
              size="sm"
              disabled={isRefreshing}
              className="bg-slate-800/90 border-blue-400/40 text-blue-300 hover:bg-slate-700/90 hover:border-blue-300/60 transition-all duration-200 backdrop-blur-md"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>

            {/* Connection Status */}
            <div className="flex items-center space-x-2 text-blue-300 bg-slate-900/90 backdrop-blur-md px-3 py-2 rounded-lg border border-blue-400/40 shadow-lg">
              {isConnected ? <Wifi className="w-4 h-4 text-green-400" /> : <WifiOff className="w-4 h-4 text-red-400" />}
              <span className="text-xs font-mono">{isConnected ? "Live" : "Offline"}</span>
            </div>
          </div>
        </div>

        {/* Bottom Stats and Info */}
        <div className="absolute bottom-4 left-4 right-4 z-10 flex items-end justify-between">
          {/* Left Side - Selected Node Info */}
          {selectedNode && (
            <div className="bg-slate-900/95 text-blue-300 p-3 rounded-lg shadow-2xl text-sm border border-blue-400/40 backdrop-blur-sm">
              <p className="font-mono font-bold">Node: {selectedNode.id}</p>
              <p className="font-mono text-xs text-blue-400">Error: {selectedNode.error.toFixed(4)}</p>
              {selectedNode.type && <p className="font-mono text-xs text-green-400">Type: {selectedNode.type}</p>}
              {selectedNode.last_active && (
                <p className="font-mono text-xs text-yellow-400">Last Active: {selectedNode.last_active}</p>
              )}
              <p className="font-mono text-xs text-purple-400">
                {is3D ? (
                  <>
                    3D Pos: ({selectedNode.position.x?.toFixed(1)}, {selectedNode.position.y?.toFixed(1)},{" "}
                    {selectedNode.position.z?.toFixed(1)})
                  </>
                ) : (
                  <>
                    2D Pos: ({selectedNode.position.x?.toFixed(1)}, {selectedNode.position.y?.toFixed(1)})
                  </>
                )}
              </p>
            </div>
          )}

          {/* Right Side - Graph Stats */}
          <div className="flex items-center space-x-2">
            <Badge
              variant="outline"
              className="bg-slate-800/90 text-slate-200 border-slate-700 font-mono shadow-lg backdrop-blur-md"
            >
              {graphData.nodes.length} nodes • {graphData.links.length} edges
            </Badge>
            <Badge
              variant="outline"
              className="bg-green-500/20 text-green-400 border-green-500/30 font-mono shadow-lg backdrop-blur-md"
            >
              {graphData.nodes.filter((n) => n.active).length} active
            </Badge>
            <Badge
              variant="outline"
              className="bg-red-500/20 text-red-400 border-red-500/30 font-mono shadow-lg backdrop-blur-md"
            >
              {graphData.nodes.filter((n) => (n.error || 0) > 0.3).length} errors
            </Badge>
          </div>
        </div>

        {/* Render appropriate visualization */}
        {is3D ? (
          <BrainVisualization3D
            networkId={networkId}
            graphData={graphData}
            selectedNode={selectedNode}
            setSelectedNode={setSelectedNode}
            activity={activity}
            isConnected={isConnected}
          />
        ) : (
          <BrainVisualization2D
            networkId={networkId}
            graphData={graphData}
            selectedNode={selectedNode}
            setSelectedNode={setSelectedNode}
            activity={activity}
            isConnected={isConnected}
            handleCanvasClick={handleCanvasClick}
          />
        )}

        {/* No Network State */}
        {!networkId && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm">
            <div className="text-center text-blue-300">
              <AlertCircle className="w-16 h-16 mx-auto mb-6 text-blue-400" />
              <p className="text-xl font-medium font-mono mb-2">NO NETWORK SELECTED</p>
              <p className="text-sm text-blue-400 font-mono">Initialize neural interface to begin</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
