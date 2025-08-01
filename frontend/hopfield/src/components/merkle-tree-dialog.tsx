"use client"

import { useEffect, useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Hash, GitBranch } from "lucide-react"
import { MerkleTree3DVisualization } from "@/components/merkle-tree-3d-visualization" // Import the 3D visualization

interface MerkleTreeDialogProps {
  isOpen: boolean
  onClose: () => void
  networkId: number
}

interface MerkleNode {
  hash: string
  level: number
  index: number
  isLeaf: boolean
  children?: MerkleNode[]
}

export function MerkleTreeDialog({ isOpen, onClose, networkId }: MerkleTreeDialogProps) {
  const [merkleTree, setMerkleTree] = useState<MerkleNode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (isOpen && networkId) {
      fetchMerkleTree()
    }
  }, [isOpen, networkId])

  const fetchMerkleTree = async () => {
    setLoading(true)
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
      <div key={level} className="flex justify-center items-center space-x-4 mb-6">
        {nodesAtLevel.map((node, index) => (
          <div key={`${level}-${index}`} className="flex flex-col items-center">
            <div
              className={`
                px-3 py-2 rounded-lg border text-xs font-mono transition-colors hover:shadow-md
                ${
                  node.isLeaf
                    ? "bg-blue-50 border-blue-200 text-blue-800 hover:bg-blue-100"
                    : "bg-purple-50 border-purple-200 text-purple-800 hover:bg-purple-100"
                }
              `}
            >
              <div className="flex items-center space-x-2">
                <Hash className="w-3 h-3" />
                <span>{node.hash.substring(0, 8)}...</span>
              </div>
            </div>

            {/* Connection lines to children */}
            {node.children && level < Math.max(...merkleTree.map((n) => n.level)) && (
              <div className="mt-2 mb-2">
                <div className="w-px h-6 bg-gray-300 mx-auto"></div>
                <div className="flex justify-center">
                  <div className="w-12 h-px bg-gray-300"></div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    )
  }

  const maxLevel = merkleTree.length > 0 ? Math.max(...merkleTree.map((node) => node.level)) : 0
  const rootNode = merkleTree.find((node) => node.level === maxLevel)

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden p-0">
        <DialogHeader className="px-6 pt-6">
          <DialogTitle className="flex items-center">
            <GitBranch className="w-5 h-5 mr-2 text-orange-600" />
            Security Hash Tree (Merkle Tree)
          </DialogTitle>
          <DialogDescription>Interactive 3D visualization of stored pattern hashes</DialogDescription>
        </DialogHeader>
        <div className="flex-grow px-6 pb-6">
          {networkId ? (
            <MerkleTree3DVisualization networkId={networkId} />
          ) : (
            <div className="text-center py-12 text-gray-500">
              <GitBranch className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium">No network selected</p>
              <p className="text-sm">Please select a network to view its Merkle tree</p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
