"use client"

import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { Badge } from "@/components/ui/badge"
import { GitBranch, Wifi, WifiOff } from "lucide-react"
import { useWebSocket } from "@/hooks/use-websocket"

interface MerkleTree3DVisualizationProps {
  networkId: number
}

interface Pattern {
  id: number
  pattern_hash: string
  usage_count: number
  created_at: string
}

interface MerkleNode {
  hash: string
  level: number
  isLeaf: boolean
  childrenHashes?: string[]
}

export function MerkleTree3DVisualization({ networkId }: MerkleTree3DVisualizationProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const controlsRef = useRef<OrbitControls>()
  const animationIdRef = useRef<number>()
  const [loading, setLoading] = useState(true)
  const [merkleNodes, setMerkleNodes] = useState<MerkleNode[]>([])

  // Use the enhanced WebSocket hook
  const { isConnected, lastMessage, connectionStatus } = useWebSocket(networkId, {
    enabled: true,
    reconnectAttempts: 5,
    reconnectInterval: 1000,
    heartbeatInterval: 30000,
    heartbeatTimeout: 35000,
  })

  const fetchPatternsAndBuildTree = async () => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${networkId}/patterns/`)
      const patterns: Pattern[] = await response.json()
      const treeNodes = buildMerkleTree(patterns)
      setMerkleNodes(treeNodes)
    } catch (error) {
      console.error("Error fetching patterns for Merkle tree:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPatternsAndBuildTree()
  }, [networkId]) // Re-fetch patterns and rebuild tree when networkId changes

  useEffect(() => {
    if (
      lastMessage &&
      (lastMessage.event_type === "pattern_stored" || lastMessage.event_type === "pattern_retrieved")
    ) {
      fetchPatternsAndBuildTree() // Re-fetch and rebuild tree on relevant updates
    }
  }, [lastMessage])

  useEffect(() => {
    if (!mountRef.current) return

    // Cleanup function for previous Three.js instance
    const cleanupThree = () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
      if (controlsRef.current) {
        controlsRef.current.dispose()
      }
      if (rendererRef.current && mountRef.current && mountRef.current.contains(rendererRef.current.domElement)) {
        mountRef.current.removeChild(rendererRef.current.domElement)
      }
      if (rendererRef.current) {
        rendererRef.current.dispose()
      }
      // Dispose of geometries and materials
      sceneRef.current?.children.forEach((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry.dispose()
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose())
          } else if (obj.material) {
            obj.material.dispose()
          }
        }
      })
      sceneRef.current = undefined
      rendererRef.current = undefined
      cameraRef.current = undefined
      controlsRef.current = undefined
    }

    // Perform cleanup before re-initializing
    cleanupThree()

    // Initialize Three.js scene
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })

    const container = mountRef.current
    const width = container.clientWidth
    const height = Math.min(width * 0.75, 400) // Smaller height for Merkle tree

    renderer.setSize(width, height)
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.enableZoom = true
    controls.enableRotate = true
    controls.enablePan = true
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.5
    controls.listenToKeyEvents(window) // Enable keyboard controls

    sceneRef.current = scene
    rendererRef.current = renderer
    cameraRef.current = camera
    controlsRef.current = controls

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(1, 1, 1)
    scene.add(directionalLight)

    // Add a GridHelper for debugging
    const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x888888)
    scene.add(gridHelper)

    // Position camera
    camera.position.set(0, 0, 10)
    camera.lookAt(0, 0, 0)

    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate)
      if (controls) controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!mountRef.current) return
      const newWidth = mountRef.current.clientWidth
      const newHeight = Math.min(newWidth * 0.75, 400)
      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }
    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      cleanupThree() // Ensure cleanup on unmount or re-render
    }
  }, [networkId]) // Dependency array includes networkId to re-initialize when it changes

  useEffect(() => {
    if (sceneRef.current) {
      // Clear previous nodes and connections from the scene, but keep the GridHelper
      const objectsToRemove = sceneRef.current.children.filter(
        (obj) => (obj instanceof THREE.Mesh || obj instanceof THREE.Line) && !(obj instanceof THREE.GridHelper),
      )
      objectsToRemove.forEach((obj) => {
        sceneRef.current?.remove(obj)
        if (obj instanceof THREE.Mesh) obj.geometry.dispose()
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => m.dispose())
        } else if (obj.material) {
          obj.material.dispose()
        }
      })

      if (merkleNodes.length > 0) {
        renderMerkleTreeIn3D(sceneRef.current, merkleNodes)
      }
    }
  }, [merkleNodes]) // Re-render 3D tree when merkleNodes data changes

  const buildMerkleTree = (patterns: Pattern[]): MerkleNode[] => {
    if (patterns.length === 0) return []

    let currentLevelNodes: MerkleNode[] = patterns.map((pattern) => ({
      hash: pattern.pattern_hash,
      level: 0,
      isLeaf: true,
    }))

    const allNodes: MerkleNode[] = [...currentLevelNodes]
    let level = 1

    while (currentLevelNodes.length > 1) {
      const nextLevelNodes: MerkleNode[] = []
      for (let i = 0; i < currentLevelNodes.length; i += 2) {
        const left = currentLevelNodes[i]
        const right = currentLevelNodes[i + 1] || currentLevelNodes[i] // Duplicate if odd number

        const combinedHash = hashCombine(left.hash, right.hash)

        const parentNode: MerkleNode = {
          hash: combinedHash,
          level,
          isLeaf: false,
          childrenHashes: [left.hash, right.hash],
        }
        nextLevelNodes.push(parentNode)
      }
      allNodes.push(...nextLevelNodes)
      currentLevelNodes = nextLevelNodes
      level++
    }
    return allNodes
  }

  const hashCombine = (hash1: string, hash2: string): string => {
    // Simple hash combination simulation for visualization purposes
    const combined = hash1 + hash2
    let hash = 0
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i)
      hash = (hash << 5) - hash + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, "0")
  }

  const renderMerkleTreeIn3D = (scene: THREE.Scene, nodes: MerkleNode[]) => {
    const nodeMap = new Map<string, THREE.Vector3>()
    const maxLevel = Math.max(...nodes.map((n) => n.level))
    const levelHeight = 2.5 // Vertical spacing between levels

    nodes.forEach((node) => {
      const level = node.level
      // Adjust radius and angle for better distribution
      const nodesAtThisLevel = nodes.filter((n) => n.level === level)
      const nodeIndexAtLevel = nodesAtThisLevel.findIndex((n) => n.hash === node.hash)
      const angle = (nodeIndexAtLevel / nodesAtThisLevel.length) * Math.PI * 2

      const radius = (maxLevel - level + 1) * 1.5 // Larger radius for lower levels
      const x = radius * Math.cos(angle)
      const y = (level - maxLevel / 2) * levelHeight // Center vertically
      const z = radius * Math.sin(angle)

      const position = new THREE.Vector3(x, y, z)
      nodeMap.set(node.hash, position)

      const nodeGeometry = new THREE.SphereGeometry(0.2, 16, 16)
      const nodeMaterial = new THREE.MeshLambertMaterial({
        color: node.isLeaf ? 0x4299e1 : 0x805ad5, // Blue for leaves, purple for internal
        emissive: node.isLeaf ? 0x2b6cb0 : 0x6b46c1,
        emissiveIntensity: 0.3,
      })
      const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial)
      mesh.position.copy(position)
      scene.add(mesh)
    })

    // Add connections
    nodes.forEach((node) => {
      if (node.childrenHashes && node.childrenHashes.length > 0) {
        const parentPos = nodeMap.get(node.hash)
        if (parentPos) {
          node.childrenHashes.forEach((childHash) => {
            const childPos = nodeMap.get(childHash)
            if (childPos) {
              const geometry = new THREE.BufferGeometry().setFromPoints([parentPos, childPos])
              const material = new THREE.LineBasicMaterial({
                color: 0x4a5568, // Gray
                transparent: true,
                opacity: 0.5,
              })
              const line = new THREE.Line(geometry, material)
              scene.add(line)
            }
          })
        }
      }
    })
  }

  if (loading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500"></div>
      </div>
    )
  }

  if (merkleNodes.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <GitBranch className="w-12 h-12 mx-auto mb-4 text-gray-300" />
        <p>No patterns stored yet</p>
        <p className="text-sm">Store some patterns to see the Merkle tree</p>
      </div>
    )
  }

  return (
    <div className="relative">
      <div
        ref={mountRef}
        className="w-full h-[400px] bg-gradient-to-br from-gray-900 to-blue-900 rounded-lg overflow-hidden"
      />
      <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/50 text-white px-3 py-2 rounded-lg backdrop-blur-sm">
        <Badge variant="outline" className="text-white border-white/50">
          {merkleNodes.filter((node) => node.isLeaf).length} patterns
        </Badge>
        <div className="flex items-center space-x-1">
          {isConnected ? <Wifi className="w-3 h-3 text-green-500" /> : <WifiOff className="w-3 h-3 text-red-500" />}
          <span className="text-xs text-gray-300 capitalize">{connectionStatus}</span>
        </div>
      </div>
      <div className="absolute bottom-4 right-4 bg-black/50 text-white px-3 py-2 rounded-lg backdrop-blur-sm">
        <p className="text-xs">Click and drag to rotate • Scroll to zoom • Auto-rotating</p>
      </div>
    </div>
  )
}
