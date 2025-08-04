"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Gamepad2,
  Play,
  Square,
  BarChart3,
  Trophy,
  Target,
  Zap,
  Eye,
  Activity,
  Loader2,
  Wifi,
  WifiOff,
} from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { useWebSocket } from "@/hooks/use-websocket"

interface GameTrainingPanelProps {
  networkId: number
}

interface TrainingResults {
  avg_reward: number
  success_rate: number
  rewards: number[]
  successes: boolean[]
}

interface EnvironmentInfo {
  name: string
  description: string
  difficulty: "easy" | "medium" | "hard"
  recommended_episodes: number
}

interface GameEvent {
  event_type: string
  data: {
    action?: string
    reward?: number
    shaped_reward?: number
    total_reward?: number
    step?: number
    episode?: number
    observation?: number[][]
    done?: boolean
  }
}

interface TrainingProgress {
  episode: number
  reward: number
  steps: number
  success: boolean
  epsilon?: number
}

export function GameTrainingPanel({ networkId }: GameTrainingPanelProps) {
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>("")
  const [episodes, setEpisodes] = useState<number>(50)
  const [isTraining, setIsTraining] = useState(false)
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [currentEpisode, setCurrentEpisode] = useState(0)
  const [evaluationResults, setEvaluationResults] = useState<TrainingResults | null>(null)
  const [availableEnvironments, setAvailableEnvironments] = useState<EnvironmentInfo[]>([])

  // Game visualization state
  const [gameState, setGameState] = useState<number[][] | null>(null)
  const [currentAction, setCurrentAction] = useState<string>("")
  const [currentReward, setCurrentReward] = useState<number>(0)
  const [totalReward, setTotalReward] = useState<number>(0)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [gameEvents, setGameEvents] = useState<GameEvent[]>([])
  const [recentStats, setRecentStats] = useState<TrainingProgress[]>([])

  const gameEventsRef = useRef<HTMLDivElement>(null)

  // WebSocket connection for real-time updates
  const { isConnected, sendMessage, lastMessage } = useWebSocket(networkId, {
    enabled: networkId !== null,
  })

  // Environment configurations
  const environmentConfigs: EnvironmentInfo[] = [
    {
      name: "gridworld",
      description: "Simple grid navigation with obstacles - reach the goal while avoiding obstacles",
      difficulty: "easy",
      recommended_episodes: 30,
    },
    {
      name: "maze",
      description: "Navigate through a randomly generated maze to reach the exit",
      difficulty: "medium",
      recommended_episodes: 50,
    },
    {
      name: "snake",
      description: "Classic snake game - eat food and grow without hitting walls or yourself",
      difficulty: "hard",
      recommended_episodes: 75,
    },
  ]

  // Load available environments on component mount
  useEffect(() => {
    setAvailableEnvironments(environmentConfigs)
    if (environmentConfigs.length > 0) {
      setSelectedEnvironment(environmentConfigs[0].name)
      setEpisodes(environmentConfigs[0].recommended_episodes)
    }
  }, [])

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    console.log("Game training received message:", lastMessage)

    switch (lastMessage.type) {
      case "training_started":
        setIsTraining(true)
        setTrainingProgress(0)
        setCurrentEpisode(0)
        setGameEvents([])
        setRecentStats([])
        toast({
          title: "Training Started",
          description: `Started training on ${lastMessage.environment} for ${lastMessage.episodes} episodes.`,
        })
        break

      case "training_progress":
        const stats = lastMessage.stats as TrainingProgress
        setCurrentEpisode(stats.episode || 0)
        setTrainingProgress(((stats.episode || 0) / episodes) * 100)
        setRecentStats((prev) => [...prev.slice(-9), stats])
        break

      case "training_completed":
        setIsTraining(false)
        setTrainingProgress(100)
        toast({
          title: "Training Complete",
          description: `Finished training on ${lastMessage.environment}.`,
        })
        break

      case "training_stopped":
        setIsTraining(false)
        toast({
          title: "Training Stopped",
          description: "Training has been stopped.",
        })
        break

      case "training_error":
        setIsTraining(false)
        toast({
          title: "Training Error",
          description: lastMessage.message,
          variant: "destructive",
        })
        break

      case "evaluation_completed":
        setIsEvaluating(false)
        setEvaluationResults(lastMessage.results)
        toast({
          title: "Evaluation Complete",
          description: `Evaluated performance on ${lastMessage.environment}.`,
        })
        break

      case "game_event":
        const gameEvent = lastMessage.payload as GameEvent
        handleGameEvent(gameEvent)
        break

      case "curriculum_started":
        setIsTraining(true)
        toast({
          title: "Curriculum Started",
          description: "Started curriculum training across all environments.",
        })
        break

      case "curriculum_completed":
        setIsTraining(false)
        toast({
          title: "Curriculum Complete",
          description: "Curriculum training completed successfully.",
        })
        break
    }
  }, [lastMessage, episodes])

  const handleGameEvent = (event: GameEvent) => {
    const { event_type, data } = event

    switch (event_type) {
      case "action":
        setCurrentAction(data.action || "")
        setCurrentStep(data.step || 0)
        break

      case "reward":
        setCurrentReward(data.reward || 0)
        setTotalReward(data.total_reward || 0)
        break

      case "observation":
        if (data.observation) {
          setGameState(data.observation)
        }
        break
    }

    // Add to events log
    setGameEvents((prev) => [...prev.slice(-49), event])

    // Auto-scroll events
    setTimeout(() => {
      if (gameEventsRef.current) {
        gameEventsRef.current.scrollTop = gameEventsRef.current.scrollHeight
      }
    }, 100)
  }

  const startTraining = async () => {
    if (!selectedEnvironment) {
      toast({
        title: "Environment Required",
        description: "Please select a game environment to train on.",
        variant: "destructive",
      })
      return
    }

    if (!isConnected) {
      toast({
        title: "Connection Error",
        description: "WebSocket not connected. Please check your connection.",
        variant: "destructive",
      })
      return
    }

    // Reset state
    setGameState(null)
    setCurrentAction("")
    setCurrentReward(0)
    setTotalReward(0)
    setCurrentStep(0)
    setGameEvents([])
    setRecentStats([])

    // Send training command via WebSocket
    sendMessage({
      type: "start_training",
      environment: selectedEnvironment,
      episodes: episodes,
      training_type: "game",
    })
  }

  const stopTraining = () => {
    if (isConnected) {
      sendMessage({ type: "stop_training" })
    }
  }

  const evaluatePerformance = async () => {
    if (!selectedEnvironment) {
      toast({
        title: "Environment Required",
        description: "Please select a game environment to evaluate on.",
        variant: "destructive",
      })
      return
    }

    if (!isConnected) {
      toast({
        title: "Connection Error",
        description: "WebSocket not connected. Please check your connection.",
        variant: "destructive",
      })
      return
    }

    setIsEvaluating(true)
    sendMessage({
      type: "evaluate_performance",
      environment: selectedEnvironment,
      episodes: 10,
    })
  }

  const runCurriculum = async () => {
    if (!isConnected) {
      toast({
        title: "Connection Error",
        description: "WebSocket not connected. Please check your connection.",
        variant: "destructive",
      })
      return
    }

    sendMessage({ type: "run_curriculum" })
  }

  const handleEnvironmentChange = (envName: string) => {
    setSelectedEnvironment(envName)
    const envConfig = availableEnvironments.find((env) => env.name === envName)
    if (envConfig) {
      setEpisodes(envConfig.recommended_episodes)
    }
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "easy":
        return "bg-green-100 text-green-800"
      case "medium":
        return "bg-yellow-100 text-yellow-800"
      case "hard":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const renderGameState = () => {
    if (!gameState) return null

    return (
      <div className="grid gap-1 p-2 bg-slate-900 rounded font-mono text-xs">
        {gameState.map((row, i) => (
          <div key={i} className="flex gap-1">
            {row.map((cell, j) => {
              let cellClass = "w-4 h-4 flex items-center justify-center rounded-sm text-xs font-bold"
              let content = ""

              if (cell === 0.5) {
                // Player
                cellClass += " bg-blue-500 text-white"
                content = "P"
              } else if (cell === 1) {
                // Goal/Food
                cellClass += " bg-green-500 text-white"
                content = "G"
              } else if (cell === -1) {
                // Obstacle/Wall
                cellClass += " bg-red-500 text-white"
                content = "█"
              } else if (cell === -0.5) {
                // Special (like maze exit)
                cellClass += " bg-yellow-500 text-black"
                content = "E"
              } else {
                // Empty
                cellClass += " bg-slate-700"
                content = "·"
              }

              return (
                <div key={j} className={cellClass}>
                  {content}
                </div>
              )
            })}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Main Training Controls */}
      <Card className="shadow-lg flex-shrink-0">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <Gamepad2 className="w-5 h-5 mr-2 text-blue-600" />
              Game Environment Training
            </div>
            {isConnected ? (
              <Badge variant="secondary" className="bg-green-100 text-green-800">
                <Wifi className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="secondary" className="bg-red-100 text-red-800">
                <WifiOff className="w-3 h-3 mr-1" />
                Disconnected
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Environment Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="environment">Game Environment</Label>
              <Select value={selectedEnvironment} onValueChange={handleEnvironmentChange} disabled={isTraining}>
                <SelectTrigger>
                  <SelectValue placeholder="Select environment" />
                </SelectTrigger>
                <SelectContent>
                  {availableEnvironments.map((env) => (
                    <SelectItem key={env.name} value={env.name}>
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="font-medium capitalize">{env.name}</span>
                          <Badge variant="secondary" className={getDifficultyColor(env.difficulty)}>
                            {env.difficulty}
                          </Badge>
                        </div>
                        <span className="text-xs text-gray-500">{env.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="episodes">Training Episodes</Label>
              <Input
                id="episodes"
                type="number"
                value={episodes}
                onChange={(e) => setEpisodes(Number(e.target.value))}
                min={10}
                max={500}
                step={10}
                disabled={isTraining}
              />
            </div>
          </div>

          {/* Training Controls */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-green-600" />
              <span className="font-medium text-sm">Training Controls</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <Button
                onClick={startTraining}
                disabled={isTraining || !selectedEnvironment || !isConnected}
                className="bg-green-600 hover:bg-green-700 disabled:opacity-50"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Training
                  </>
                )}
              </Button>

              <Button
                onClick={evaluatePerformance}
                disabled={isEvaluating || !selectedEnvironment || !isConnected}
                variant="outline"
              >
                {isEvaluating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Evaluating...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Evaluate
                  </>
                )}
              </Button>

              <Button
                onClick={runCurriculum}
                disabled={isTraining || !isConnected}
                variant="outline"
                className="bg-purple-50 hover:bg-purple-100"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Target className="w-4 h-4 mr-2" />
                    Full Curriculum
                  </>
                )}
              </Button>
            </div>

            {isTraining && (
              <>
                <Button
                  onClick={stopTraining}
                  variant="outline"
                  className="w-full bg-red-50 hover:bg-red-100"
                  disabled={!isConnected}
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </Button>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Training Progress</span>
                    <span>
                      {Math.round(trainingProgress)}% (Episode {currentEpisode}/{episodes})
                    </span>
                  </div>
                  <Progress value={trainingProgress} className="w-full" />
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Real-time Game Visualization - Flex grow to fill remaining space */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Game State Visualization */}
        <Card className="shadow-lg h-full flex flex-col">
          <CardHeader className="flex-shrink-0">
            <CardTitle className="flex items-center">
              <Eye className="w-5 h-5 mr-2 text-purple-600" />
              Game State
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col space-y-4">
            {gameState ? (
              <div className="flex-1 flex items-center justify-center">{renderGameState()}</div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-center text-muted-foreground">
                <div>
                  <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Game visualization will appear here during training</p>
                </div>
              </div>
            )}
            {/* Current Game Stats */}
            <div className="grid grid-cols-2 gap-4 text-sm flex-shrink-0">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Step:</span>
                  <Badge variant="outline">{currentStep}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Action:</span>
                  <Badge variant="secondary">{currentAction || "None"}</Badge>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Reward:</span>
                  <Badge variant={currentReward > 0 ? "default" : "destructive"}>{currentReward.toFixed(2)}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Total:</span>
                  <Badge variant="outline">{totalReward.toFixed(2)}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Events Log */}
        <Card className="shadow-lg h-full flex flex-col">
          <CardHeader className="flex-shrink-0">
            <CardTitle className="flex items-center">
              <Activity className="w-5 h-5 mr-2 text-orange-600" />
              Game Events
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            <ScrollArea className="h-full">
              <div ref={gameEventsRef} className="space-y-2">
                {gameEvents.length > 0 ? (
                  gameEvents.map((event, index) => (
                    <div key={index} className="text-xs p-2 bg-slate-50 rounded border-l-2 border-blue-200">
                      <div className="flex justify-between items-center">
                        <span className="font-medium text-blue-600">{event.event_type}</span>
                        <span className="text-muted-foreground">Step {event.data.step || 0}</span>
                      </div>
                      {event.data.action && <div className="text-slate-600">Action: {event.data.action}</div>}
                      {event.data.reward !== undefined && (
                        <div className={`${event.data.reward > 0 ? "text-green-600" : "text-red-600"}`}>
                          Reward: {event.data.reward.toFixed(2)}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="h-full flex items-center justify-center text-center text-muted-foreground">
                    <div>
                      <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>Game events will appear here during training</p>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Recent Training Stats */}
      {recentStats.length > 0 && (
        <Card className="shadow-lg flex-shrink-0">
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-indigo-600" />
              Recent Episodes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              {recentStats.slice(-5).map((stat, index) => (
                <div key={index} className="text-center p-3 bg-slate-50 rounded">
                  <div className="text-lg font-bold text-indigo-600">#{stat.episode}</div>
                  <div className="text-sm text-muted-foreground">Reward: {stat.reward.toFixed(1)}</div>
                  <div className="text-xs text-muted-foreground">Steps: {stat.steps}</div>
                  <Badge
                    variant={stat.success ? "default" : "secondary"}
                    className={stat.success ? "bg-green-100 text-green-800" : ""}
                  >
                    {stat.success ? "Success" : "Failed"}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Evaluation Results */}
      {evaluationResults && (
        <Card className="shadow-lg flex-shrink-0">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Trophy className="w-5 h-5 mr-2 text-yellow-600" />
              Performance Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{evaluationResults.avg_reward.toFixed(1)}</div>
                <div className="text-sm text-gray-600">Avg Reward</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {(evaluationResults.success_rate * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Success Rate</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {Math.max(...evaluationResults.rewards).toFixed(1)}
                </div>
                <div className="text-sm text-gray-600">Best Score</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{evaluationResults.rewards.length}</div>
                <div className="text-sm text-gray-600">Episodes</div>
              </div>
            </div>

            <div className="mt-4 flex justify-center">
              <Badge
                variant={evaluationResults.success_rate > 0.7 ? "default" : "secondary"}
                className={evaluationResults.success_rate > 0.7 ? "bg-green-100 text-green-800" : ""}
              >
                {evaluationResults.success_rate > 0.7
                  ? "Excellent Performance"
                  : evaluationResults.success_rate > 0.4
                    ? "Good Performance"
                    : "Needs Improvement"}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
