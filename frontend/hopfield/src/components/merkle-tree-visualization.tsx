"use client"

import { useEffect, useState } from "react"
import { Hash, GitBranch } from "lucide-react"

interface MerkleTreeVisualizationProps {
  networkId: number
}

interface MerkleNode {
  hash: string
  level: number
  index: number
  isLeaf: boolean
  children?: MerkleNode[]
}

export function MerkleTreeVisualization({ networkId }: MerkleTreeVisualizationProps) {
  const [merkleTree, setMerkleTree] = useState<MerkleNode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMerkleTree()
    const interval = setInterval(fetchMerkleTree, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [networkId])

  const fetchMerkleTree = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${networkId}/patterns/`)
      const patterns = await response.json()

      if (patterns.length > 0) {
        const tree = buildMerkleTree(patterns)
        setMerkleTree(tree)
      } else {
        setMerkleTree([])
      }
    } catch (error) {
      console.error("Error fetching merkle tree:", error)
    } finally {
      setLoading(false)
    }
  }

  const buildMerkleTree = (patterns: any[]): MerkleNode[] => {
    if (patterns.length === 0) return []

    // Create leaf nodes
    let currentLevel = patterns.map((pattern, index) => ({
      hash: pattern.pattern_hash,
      level: 0,
      index,
      isLeaf: true,
    }))

    const allNodes: MerkleNode[] = [...currentLevel]
    let level = 1

    // Build tree bottom-up
    while (currentLevel.length > 1) {
      const nextLevel: MerkleNode[] = []

      // Pair up nodes and create parent nodes
      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i]
        const right = currentLevel[i + 1] || currentLevel[i] // Duplicate if odd number

        // Simulate hash combination
        const combinedHash = hashCombine(left.hash, right.hash)

        const parentNode: MerkleNode = {
          hash: combinedHash,
          level,
          index: Math.floor(i / 2),
          isLeaf: false,
          children: [left, right],
        }

        nextLevel.push(parentNode)
      }

      allNodes.push(...nextLevel)
      currentLevel = nextLevel
      level++
    }

    return allNodes
  }

  const hashCombine = (hash1: string, hash2: string): string => {
    // Simple hash combination simulation
    const combined = hash1 + hash2
    let hash = 0
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i)
      hash = (hash << 5) - hash + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).padStart(8, "0")
  }

  const renderTreeLevel = (level: number) => {
    const nodesAtLevel = merkleTree.filter((node) => node.level === level)
    if (nodesAtLevel.length === 0) return null

    return (
      <div key={level} className="flex justify-center items-center space-x-4 mb-2">
        {nodesAtLevel.map((node, index) => (
          <div key={`${level}-${index}`} className="flex flex-col items-center">
            <div
              className={`
                px-2 py-1 rounded text-xs font-mono border
                ${
                  node.isLeaf
                    ? "bg-blue-500/20 border-blue-400/50 text-blue-200"
                    : "bg-purple-500/20 border-purple-400/50 text-purple-200"
                }
              `}
            >
              <div className="flex items-center space-x-1">
                <Hash className="w-3 h-3" />
                <span>{node.hash.substring(0, 8)}...</span>
              </div>
            </div>

            {/* Connection lines to children */}
            {node.children && (
              <div className="mt-2 mb-2">
                <div className="w-px h-4 bg-gray-300 mx-auto"></div>
                <div className="flex justify-center">
                  <div className="w-8 h-px bg-gray-300"></div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="h-full">
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      </div>
    )
  }

  if (merkleTree.length === 0) {
    return (
      <div className="text-center py-4 text-blue-200">
        <GitBranch className="w-8 h-8 mx-auto mb-2 text-blue-400" />
        <p className="text-sm">No patterns stored yet</p>
      </div>
    )
  }

  const maxLevel = Math.max(...merkleTree.map((node) => node.level))
  const rootNode = merkleTree.find((node) => node.level === maxLevel)

  return (
    <div className="h-full">
      <div className="overflow-x-auto h-full">
        <div className="min-w-max py-2 h-full flex flex-col justify-center">
          {/* Render tree from root to leaves */}
          {Array.from({ length: maxLevel + 1 }, (_, i) => maxLevel - i).map((level) => renderTreeLevel(level))}
        </div>
      </div>

      {rootNode && (
        <div className="mt-2 p-2 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-blue-200">Root:</span>
            <span className="font-mono text-blue-300">{rootNode.hash.substring(0, 12)}...</span>
          </div>
        </div>
      )}
    </div>
  )
}
