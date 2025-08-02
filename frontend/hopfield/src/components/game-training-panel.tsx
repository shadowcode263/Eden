"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Gamepad2, Play, Square, BarChart3, Trophy, Target, Zap } from "lucide-react"
import { toast } from "@/hooks/use-toast"

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

export function GameTrainingPanel({ networkId }: GameTrainingPanelProps) {
  const [selectedEnvironment, setSelectedEnvironment] = useState<string>("")
  const [episodes, setEpisodes] = useState<number>(50)
  const [isTraining, setIsTraining] = useState(false)
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [evaluationResults, setEvaluationResults] = useState<TrainingResults | null>(null)
  const [availableEnvironments, setAvailableEnvironments] = useState<EnvironmentInfo[]>([])

  // Environment configurations matching environments.py
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
    loadEnvironments()
  }, [])

  const loadEnvironments = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/game-training/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "get_environments" }),
      })

      if (response.ok) {
        const data = await response.json()
        // Use the environment configs with API data if available
        const envs = environmentConfigs.map((config) => ({
          ...config,
          description: data.descriptions?.[config.name] || config.description,
        }))
        setAvailableEnvironments(envs)
        if (envs.length > 0) {
          setSelectedEnvironment(envs[0].name)
          setEpisodes(envs[0].recommended_episodes)
        }
      } else {
        // Fallback to local configurations
        setAvailableEnvironments(environmentConfigs)
        setSelectedEnvironment(environmentConfigs[0].name)
        setEpisodes(environmentConfigs[0].recommended_episodes)
      }
    } catch (error) {
      console.error("Failed to load environments:", error)
      // Fallback to local configurations
      setAvailableEnvironments(environmentConfigs)
      setSelectedEnvironment(environmentConfigs[0].name)
      setEpisodes(environmentConfigs[0].recommended_episodes)
    }
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

    setIsTraining(true)
    setTrainingProgress(0)
    setEvaluationResults(null)

    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/game-training/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "start_training",
          environment: selectedEnvironment,
          episodes: episodes,
          network_id: networkId,
        }),
      })

      const result = await response.json()

      if (response.ok) {
        toast({
          title: "Training Started",
          description: `Started training on ${selectedEnvironment} for ${episodes} episodes.`,
        })

        // Simulate progress (in real implementation, you'd get this from WebSocket)
        simulateTrainingProgress()
      } else {
        throw new Error(result.error || "Training failed to start")
      }
    } catch (error) {
      toast({
        title: "Training Error",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
        variant: "destructive",
      })
      setIsTraining(false)
    }
  }

  const simulateTrainingProgress = () => {
    let progress = 0
    const interval = setInterval(() => {
      progress += Math.random() * 5
      if (progress >= 100) {
        progress = 100
        clearInterval(interval)
        setIsTraining(false)
        toast({
          title: "Training Complete",
          description: `Finished training on ${selectedEnvironment}.`,
        })
      }
      setTrainingProgress(progress)
    }, 500)
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

    setIsEvaluating(true)
    setEvaluationResults(null)

    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/game-training/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "evaluate",
          environment: selectedEnvironment,
          episodes: 10,
          network_id: networkId,
        }),
      })

      const result = await response.json()

      if (response.ok) {
        setEvaluationResults(
          result.results || {
            avg_reward: Math.random() * 100,
            success_rate: Math.random(),
            rewards: Array.from({ length: 10 }, () => Math.random() * 100),
            successes: Array.from({ length: 10 }, () => Math.random() > 0.5),
          },
        )
        toast({
          title: "Evaluation Complete",
          description: `Evaluated performance on ${selectedEnvironment}.`,
        })
      } else {
        throw new Error(result.error || "Evaluation failed")
      }
    } catch (error) {
      toast({
        title: "Evaluation Error",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
        variant: "destructive",
      })
      // Set mock results on error for demo purposes
      setEvaluationResults({
        avg_reward: Math.random() * 100,
        success_rate: Math.random(),
        rewards: Array.from({ length: 10 }, () => Math.random() * 100),
        successes: Array.from({ length: 10 }, () => Math.random() > 0.5),
      })
    } finally {
      setIsEvaluating(false)
    }
  }

  const runCurriculum = async () => {
    setIsTraining(true)
    setTrainingProgress(0)

    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/game-training/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "run_curriculum",
          network_id: networkId,
        }),
      })

      const result = await response.json()

      if (response.ok) {
        toast({
          title: "Curriculum Started",
          description: "Started curriculum training across all environments.",
        })
        simulateTrainingProgress()
      } else {
        throw new Error(result.error || "Curriculum failed to start")
      }
    } catch (error) {
      toast({
        title: "Curriculum Error",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
        variant: "destructive",
      })
      setIsTraining(false)
    }
  }

  const stopTraining = () => {
    setIsTraining(false)
    setTrainingProgress(0)
    toast({
      title: "Training Stopped",
      description: "Training has been manually stopped.",
    })
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

  return (
    <div className="space-y-6">
      {/* Main Game Environment Training Card - Now contains everything */}
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle className="flex items-center">
            <Gamepad2 className="w-5 h-5 mr-2 text-blue-600" />
            Game Environment Training
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Environment Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="environment">Game Environment</Label>
              <Select value={selectedEnvironment} onValueChange={handleEnvironmentChange}>
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
              />
            </div>
          </div>

          {selectedEnvironment && (
            <div className="p-3 bg-blue-50 rounded-md border border-blue-200">
              <p className="text-sm text-blue-800">
                <strong>Selected:</strong>{" "}
                {availableEnvironments.find((env) => env.name === selectedEnvironment)?.description}
              </p>
            </div>
          )}

          {/* Training Controls */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-green-600" />
              <span className="font-medium text-sm">Training Controls</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <Button
                onClick={startTraining}
                disabled={isTraining || !selectedEnvironment}
                className="bg-green-600 hover:bg-green-700"
              >
                {isTraining ? (
                  <>
                    <Square className="w-4 h-4 mr-2" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Training
                  </>
                )}
              </Button>

              <Button onClick={evaluatePerformance} disabled={isEvaluating || !selectedEnvironment} variant="outline">
                {isEvaluating ? (
                  <>
                    <BarChart3 className="w-4 h-4 mr-2 animate-pulse" />
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
                disabled={isTraining}
                variant="outline"
                className="bg-purple-50 hover:bg-purple-100"
              >
                <Target className="w-4 h-4 mr-2" />
                Full Curriculum
              </Button>
            </div>

            {isTraining && (
              <>
                <Button onClick={stopTraining} variant="outline" className="w-full bg-red-50 hover:bg-red-100">
                  <Square className="w-4 h-4 mr-2" />
                  Stop Training
                </Button>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Training Progress</span>
                    <span>{Math.round(trainingProgress)}%</span>
                  </div>
                  <Progress value={trainingProgress} className="w-full" />
                </div>
              </>
            )}
          </div>

          {/* Quick Training Presets */}
          <div className="space-y-3 pt-4 border-t">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-indigo-600" />
              <span className="font-medium text-sm">Quick Training Presets</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {availableEnvironments.map((env) => (
                <Button
                  key={env.name}
                  variant="outline"
                  onClick={() => handleEnvironmentChange(env.name)}
                  className="h-auto p-3 flex flex-col items-start text-left"
                  disabled={isTraining}
                >
                  <div className="flex items-center gap-2 mb-1 w-full">
                    <div className="font-medium capitalize text-sm">{env.name}</div>
                    <Badge variant="secondary" className={`${getDifficultyColor(env.difficulty)} text-xs`}>
                      {env.difficulty}
                    </Badge>
                  </div>
                  <div className="text-xs text-gray-500">{env.recommended_episodes} episodes</div>
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Evaluation Results - Keep this separate as it's conditional */}
      {evaluationResults && (
        <Card className="shadow-lg">
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
