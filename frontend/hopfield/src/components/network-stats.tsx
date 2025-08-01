"use client"

import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, Activity } from "lucide-react"
import { useWebSocket } from "@/hooks/use-websocket"

interface NetworkStatsProps {
  network: {
    id: number
    name: string
    embedding_dim: number
    beta: number
    learning_rate: number
    total_patterns: number
    total_retrievals: number
  }
}

interface NetworkStatsData {
  total_patterns: number
  total_retrievals: number
  embedding_dim: number
  merkle_root: string | null
  most_used_patterns: Array<{
    hash: string
    usage_count: number
  }>
}

export function NetworkStats({ network }: NetworkStatsProps) {
  const [stats, setStats] = useState<NetworkStatsData | null>(null)
  const [loading, setLoading] = useState(true)

  // Use the new WebSocket hook
  const { lastMessage } = useWebSocket(network.id)

  const fetchStats = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${network.id}/stats/`)
      const data = await response.json()
      setStats(data)
      console.log("Fetched Network Stats:", data)
    } catch (error) {
      console.error("Error fetching stats:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    console.log("NetworkStats: networkId changed to", network.id)
    fetchStats()
  }, [network.id])

  // Listen for WebSocket updates to refresh stats
  useEffect(() => {
    if (
      lastMessage &&
      (lastMessage.event_type === "pattern_stored" || lastMessage.event_type === "pattern_retrieved")
    ) {
      console.log("NetworkStats: WebSocket update received, re-fetching stats.")
      fetchStats()
    }
  }, [lastMessage])

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Basic Stats */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="bg-blue-50 border-blue-200">
          <CardContent className="p-4 text-center">
            <Database className="w-6 h-6 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-blue-700">{stats?.total_patterns || 0}</div>
            <div className="text-sm text-blue-600">Patterns</div>
          </CardContent>
        </Card>

        <Card className="bg-green-50 border-green-200">
          <CardContent className="p-4 text-center">
            <Activity className="w-6 h-6 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-green-700">{stats?.total_retrievals || 0}</div>
            <div className="text-sm text-green-600">Retrievals</div>
          </CardContent>
        </Card>
      </div>

      {/* Most Used Patterns */}
      {stats?.most_used_patterns && stats.most_used_patterns.length > 0 && (
        <Card className="bg-gray-50 border-gray-200">
          <CardContent className="p-4">
            <div className="font-medium mb-3 text-gray-900">Most Active Patterns</div>
            <div className="space-y-2">
              {stats.most_used_patterns.map((pattern, index) => (
                <div key={pattern.hash} className="flex justify-between items-center text-sm">
                  <span className="font-mono text-gray-600">{pattern.hash.substring(0, 8)}...</span>
                  <Badge variant="secondary" className="bg-gray-200 text-gray-700">
                    {pattern.usage_count} uses
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
