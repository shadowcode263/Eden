"use client"

import { useCallback, useEffect, useRef, useState } from "react"

interface WebSocketOptions {
  url?: string
  enabled?: boolean
  reconnectAttempts?: number
  reconnectInterval?: number
  heartbeatInterval?: number
  heartbeatTimeout?: number
  showToasts?: boolean
}

interface WebSocketMessage {
  type: string
  event_type?: string
  data?: any
  timestamp?: number
  [key: string]: any
}

const DEFAULT_RECONNECT_INTERVAL = 1000 // 1 second
const DEFAULT_MAX_RECONNECT_ATTEMPTS = 5
const DEFAULT_HEARTBEAT_INTERVAL = 30000 // 30 seconds
const DEFAULT_HEARTBEAT_TIMEOUT = DEFAULT_HEARTBEAT_INTERVAL + 5000 // 35 seconds

export function useWebSocket(networkId: number | null, options: WebSocketOptions = {}) {
  const {
    url = "ws://localhost:8000",
    enabled = true,
    reconnectAttempts: maxReconnectAttempts = DEFAULT_MAX_RECONNECT_ATTEMPTS,
    reconnectInterval: baseReconnectInterval = DEFAULT_RECONNECT_INTERVAL,
    heartbeatInterval: wsHeartbeatInterval = DEFAULT_HEARTBEAT_INTERVAL,
    heartbeatTimeout: wsHeartbeatTimeout = DEFAULT_HEARTBEAT_TIMEOUT,
    showToasts = false,
  } = options

  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isConnectingRef = useRef(false)
  const lastHeartbeatRef = useRef<number>(Date.now())
  const connectionHealthRef = useRef<boolean>(true)
  const isManuallyClosed = useRef(false)
  const messageQueue = useRef<string[]>([])
  const reconnectAttempts = useRef(0)

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<Event | null>(null)
  const [connectionHealth, setConnectionHealth] = useState({
    isConnected: false,
    isConnecting: false,
    reconnectAttempts: 0,
    queuedMessages: 0,
    lastHeartbeat: Date.now(),
    connectionHealthy: true,
    timeSinceLastHeartbeat: 0,
  })

  // Enhanced message handler
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    try {
      const data: WebSocketMessage = JSON.parse(event.data)
      console.log("WebSocket message received:", data.type, data)

      // Handle heartbeat acknowledgment
      if (data.type === "pong" || data.type === "heartbeat_ack") {
        lastHeartbeatRef.current = Date.now()
        connectionHealthRef.current = true

        // Clear the heartbeat timeout as an ACK was received
        if (heartbeatTimeoutRef.current) {
          clearTimeout(heartbeatTimeoutRef.current)
          heartbeatTimeoutRef.current = null
        }

        console.log("Heartbeat acknowledged - connection healthy")
        updateConnectionHealth()
        return
      }

      // Handle ping from server
      if (data.type === "ping") {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: "pong", timestamp: Date.now() }))
        }
        return
      }

      // Handle brain network updates
      if (data.type === "graph_state_update" || data.type === "network_update") {
        setLastMessage(data)
        setError(null)
      }
    } catch (error: any) {
      console.error("Failed to parse WebSocket message:", error, event.data)
    }
  }, [])

  // Utility to clear the reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  // Utility to clear the heartbeat sending interval
  const clearHeartbeatInterval = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }
  }, [])

  // Utility to clear the heartbeat response timeout
  const clearHeartbeatTimeout = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current)
      heartbeatTimeoutRef.current = null
    }
  }, [])

  // Update connection health state
  const updateConnectionHealth = useCallback(() => {
    setConnectionHealth({
      isConnected: ws.current?.readyState === WebSocket.OPEN,
      isConnecting: ws.current?.readyState === WebSocket.CONNECTING,
      reconnectAttempts: reconnectAttempts.current,
      queuedMessages: messageQueue.current.length,
      lastHeartbeat: lastHeartbeatRef.current,
      connectionHealthy: connectionHealthRef.current,
      timeSinceLastHeartbeat: Date.now() - lastHeartbeatRef.current,
    })
  }, [])

  // Resets the heartbeat timeout, expecting an ACK
  const resetHeartbeat = useCallback(() => {
    clearHeartbeatTimeout()
    heartbeatTimeoutRef.current = setTimeout(() => {
      console.warn("WebSocket heartbeat timed out. Attempting to reconnect...")
      connectionHealthRef.current = false
      updateConnectionHealth()
      // Close the connection cleanly to trigger the onclose handler and reconnection logic
      if (ws.current) {
        ws.current.close(1000, "Heartbeat timeout")
      }
    }, wsHeartbeatTimeout)
  }, [clearHeartbeatTimeout, wsHeartbeatTimeout, updateConnectionHealth])

  // Starts sending heartbeats periodically
  const startHeartbeat = useCallback(() => {
    clearHeartbeatInterval()
    heartbeatIntervalRef.current = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        const heartbeatMessage = JSON.stringify({
          type: "ping",
          timestamp: Date.now(),
        })
        ws.current.send(heartbeatMessage)
        resetHeartbeat() // Start timeout for ACK after sending heartbeat
        updateConnectionHealth()
      }
    }, wsHeartbeatInterval)
  }, [clearHeartbeatInterval, wsHeartbeatInterval, resetHeartbeat, updateConnectionHealth])

  // Connect to WebSocket with enhanced stability
  const connect = useCallback(async () => {
    if (!enabled || networkId === null || isManuallyClosed.current) {
      console.log("WebSocket connection disabled or no networkId provided")
      return
    }

    // Prevent multiple concurrent connection attempts
    if (isConnectingRef.current) {
      console.log("Connection attempt already in progress, skipping...")
      return
    }

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected")
      return
    }

    if (ws.current && ws.current.readyState === WebSocket.CONNECTING) {
      console.log("WebSocket already connecting")
      return
    }

    isConnectingRef.current = true

    try {
      const wsUrl = `${url}/ws/network/${networkId}/`
      console.log("Attempting to connect to WebSocket:", wsUrl)

      // Close existing connection if any
      if (ws.current) {
        try {
          ws.current.close(1000, "Reconnecting")
        } catch (error) {
          console.warn("Error closing existing connection:", error)
        }
        ws.current = null
      }

      ws.current = new WebSocket(wsUrl)

      // Set up connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.current && ws.current.readyState === WebSocket.CONNECTING) {
          console.warn("WebSocket connection timeout")
          ws.current.close()
        }
      }, 10000) // 10 second timeout

      ws.current.onopen = (event) => {
        clearTimeout(connectionTimeout)
        console.log("WebSocket connected successfully")
        isConnectingRef.current = false
        setIsConnected(true)
        setError(null)
        lastHeartbeatRef.current = Date.now()
        connectionHealthRef.current = true
        reconnectAttempts.current = 0 // Reset reconnect attempts on successful connection

        // Send any queued messages
        const queuedMessages = [...messageQueue.current]
        messageQueue.current = []
        queuedMessages.forEach((message) => {
          if (ws.current?.readyState === WebSocket.OPEN) {
            console.log("Sending queued message:", message)
            ws.current.send(message)
          } else {
            // Re-queue if connection is no longer open
            messageQueue.current.push(message)
          }
        })

        // Setup heartbeat interval
        startHeartbeat()
        updateConnectionHealth()
      }

      ws.current.onmessage = handleWebSocketMessage

      ws.current.onclose = (event) => {
        clearHeartbeatInterval() // Stop sending heartbeats
        clearHeartbeatTimeout() // Clear any pending heartbeat timeouts
        console.log("WebSocket disconnected:", event.code, event.reason)

        isConnectingRef.current = false
        setIsConnected(false)
        connectionHealthRef.current = false
        updateConnectionHealth()

        // Attempt to reconnect only if not manually closed
        if (!isManuallyClosed.current && networkId !== null) {
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current++
            // Calculate exponential backoff delay
            const delay = Math.min(
              baseReconnectInterval * Math.pow(2, reconnectAttempts.current - 1),
              30000, // Cap max delay at 30 seconds
            )
            console.log(`Attempting to reconnect in ${delay / 1000} seconds (attempt ${reconnectAttempts.current})...`)
            reconnectTimeoutRef.current = setTimeout(connect, delay)
          } else {
            console.error("Max reconnection attempts reached. Giving up.")
          }
        }
      }

      ws.current.onerror = (event) => {
        isConnectingRef.current = false
        setIsConnected(false)
        setError(event)
        connectionHealthRef.current = false
        updateConnectionHealth()
      }
    } catch (error: any) {
      console.error("Failed to create WebSocket connection:", error)
      isConnectingRef.current = false
      setIsConnected(false)
      setError(error)
      connectionHealthRef.current = false
      updateConnectionHealth()
    }
  }, [
    enabled,
    networkId,
    url,
    handleWebSocketMessage,
    maxReconnectAttempts,
    baseReconnectInterval,
    startHeartbeat,
    updateConnectionHealth,
  ])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    isManuallyClosed.current = true // Set flag to prevent automatic reconnection
    clearReconnectTimeout()
    clearHeartbeatInterval()
    clearHeartbeatTimeout()

    if (ws.current) {
      console.log("Manually closing WebSocket.")
      ws.current.close(1000, "Manual disconnect")
      ws.current = null
    }

    setIsConnected(false)
    setLastMessage(null)
    setError(null)
    updateConnectionHealth()
  }, [clearReconnectTimeout, clearHeartbeatInterval, clearHeartbeatTimeout, updateConnectionHealth])

  // Connect/disconnect based on enabled state and networkId
  useEffect(() => {
    if (enabled && networkId !== null) {
      // Reset manual close flag when effect runs and connection is enabled
      isManuallyClosed.current = false
      setLastMessage(null) // Clear previous messages when network changes
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [enabled, networkId, connect, disconnect])

  // Send JSON message with enhanced reliability
  const sendMessage = useCallback(
    (message: WebSocketMessage) => {
      const messageStr = JSON.stringify(message)
      console.log("Attempting to send message:", messageStr)

      if (ws.current?.readyState === WebSocket.OPEN && connectionHealthRef.current) {
        console.log("Sending message immediately:", messageStr)
        try {
          ws.current.send(messageStr)
          return true
        } catch (error) {
          console.error("Error sending message:", error)
          messageQueue.current.push(messageStr)
          return false
        }
      } else {
        console.warn("WebSocket not ready, queuing message:", messageStr)
        // Prevent queue from growing too large
        if (messageQueue.current.length < 100) {
          messageQueue.current.push(messageStr)
        } else {
          console.warn("Message queue full, dropping oldest message")
          messageQueue.current.shift()
          messageQueue.current.push(messageStr)
        }

        // Try to reconnect if not already connecting
        if (!isConnectingRef.current && networkId !== null && enabled) {
          console.log("Attempting to reconnect to send queued message")
          connect()
        }
        return false
      }
    },
    [connect, networkId, enabled],
  )

  // Periodic connection health update
  useEffect(() => {
    const healthInterval = setInterval(updateConnectionHealth, 5000) // Update every 5 seconds
    return () => clearInterval(healthInterval)
  }, [updateConnectionHealth])

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    disconnect,
    connect,
    connectionHealth,
    connectionStatus: isConnected ? "connected" : isConnectingRef.current ? "connecting" : "disconnected",
  }
}
