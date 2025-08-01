"use client"

import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { formatDistanceToNow } from "date-fns"
import { useWebSocket } from "@/hooks/use-websocket"

interface PatternHistoryProps {
  networkId: number
  type: "patterns" | "retrievals"
}

interface Pattern {
  id: number
  pattern_hash: string
  text_content: string // Added text_content
  usage_count: number
  created_at: string
}

interface Retrieval {
  id: number
  query_text: string
  retrieved_pattern_hash: string
  retrieved_text: string // Added retrieved_text
  confidence_score: number
  retrieval_steps: number
  created_at: string
}

export function PatternHistory({ networkId, type }: PatternHistoryProps) {
  const [data, setData] = useState<Pattern[] | Retrieval[]>([])
  const [loading, setLoading] = useState(true)

  // Use the new WebSocket hook
  const { lastMessage } = useWebSocket(networkId)

  const fetchData = async () => {
    try {
      const endpoint = type === "patterns" ? "patterns" : "retrievals"
      const response = await fetch(`http://localhost:8000/api/networks/${networkId}/${endpoint}/`)
      const result = await response.json()
      setData(result)
      console.log(`Fetched ${type} for network ${networkId}:`, result)
    } catch (error) {
      console.error(`Error fetching ${type}:`, error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    console.log(`PatternHistory (${type}): networkId changed to`, networkId)
    fetchData()
  }, [networkId, type])

  // Listen for WebSocket updates to refresh history
  useEffect(() => {
    if (
      lastMessage &&
      (lastMessage.event_type === "pattern_stored" || lastMessage.event_type === "pattern_retrieved")
    ) {
      console.log(`PatternHistory (${type}): WebSocket update received, re-fetching data.`)
      fetchData()
    }
  }, [lastMessage])

  if (loading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-16 bg-gray-100 rounded-lg animate-pulse" />
        ))}
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No {type} found</p>
        <p className="text-sm">
          {type === "patterns" ? "Store some patterns to see them here" : "Retrieve some patterns to see the history"}
        </p>
      </div>
    )
  }

  return (
    <ScrollArea className="h-[300px]">
      <div className="space-y-3">
        {type === "patterns"
          ? (data as Pattern[]).map((pattern) => (
              <Card key={pattern.id} className="hover:shadow-md transition-shadow bg-gray-50 border-gray-200">
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div className="font-mono text-sm text-gray-600">
                      {pattern.text_content
                        ? `"${pattern.text_content}"`
                        : `${pattern.pattern_hash.substring(0, 16)}...`}
                    </div>
                    <Badge variant="secondary" className="bg-gray-200 text-gray-700">
                      {pattern.usage_count} uses
                    </Badge>
                  </div>
                  <div className="text-xs text-gray-500">
                    Stored {formatDistanceToNow(new Date(pattern.created_at), { addSuffix: true })}
                  </div>
                </CardContent>
              </Card>
            ))
          : (data as Retrieval[]).map((retrieval) => (
              <Card key={retrieval.id} className="hover:shadow-md transition-shadow bg-gray-50 border-gray-200">
                <CardContent className="p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div className="text-sm font-medium truncate pr-2">"{retrieval.query_text}"</div>
                    <Badge
                      variant={retrieval.confidence_score > 0.8 ? "default" : "secondary"}
                      className={`shrink-0 ${
                        retrieval.confidence_score > 0.8
                          ? "bg-green-100 text-green-700 border-green-300"
                          : "bg-gray-200 text-gray-700"
                      }`}
                    >
                      {(retrieval.confidence_score * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>Retrieved {formatDistanceToNow(new Date(retrieval.created_at), { addSuffix: true })}</span>
                    <span>{retrieval.retrieval_steps} steps</span>
                  </div>
                  <div className="mt-2 font-mono text-xs text-gray-400">
                    →{" "}
                    {retrieval.retrieved_text
                      ? `"${retrieval.retrieved_text}"`
                      : `${retrieval.retrieved_pattern_hash?.substring(0, 12)}...`}
                  </div>
                </CardContent>
              </Card>
            ))}
      </div>
    </ScrollArea>
  )
}
