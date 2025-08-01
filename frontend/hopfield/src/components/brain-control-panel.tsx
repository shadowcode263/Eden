"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Settings, RotateCcw, ZoomIn, ZoomOut, RotateCw, Home, Save, RefreshCw, Zap } from "lucide-react" // Added Zap and Search
import { toast } from "@/hooks/use-toast"
import { NetworkControls } from "@/components/network-controls" // Import NetworkControls

interface BrainControlPanelProps {
  network: {
    id: number
    name: string
    embedding_dim: number
    beta: number
    learning_rate: number
  }
  onNetworkUpdated: (network: any) => void
}

export function BrainControlPanel({ network, onNetworkUpdated }: BrainControlPanelProps) {
  const [embeddingDim, setEmbeddingDim] = useState([network.embedding_dim])
  const [beta, setBeta] = useState([network.beta])
  const [learningRate, setLearningRate] = useState([network.learning_rate])
  const [isUpdating, setIsUpdating] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  // Camera controls state
  const [cameraDistance, setCameraDistance] = useState([15])
  const [rotationSpeed, setRotationSpeed] = useState([0.5])
  const [autoRotate, setAutoRotate] = useState(true)

  useEffect(() => {
    setEmbeddingDim([network.embedding_dim])
    setBeta([network.beta])
    setLearningRate([network.learning_rate])
    setHasChanges(false)
  }, [network])

  useEffect(() => {
    const hasParameterChanges =
      embeddingDim[0] !== network.embedding_dim || beta[0] !== network.beta || learningRate[0] !== network.learning_rate

    setHasChanges(hasParameterChanges)
  }, [embeddingDim, beta, learningRate, network])

  const handleUpdateParameters = async () => {
    setIsUpdating(true)
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${network.id}/`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          embedding_dim: embeddingDim[0],
          beta: beta[0],
          learning_rate: learningRate[0],
        }),
      })

      if (response.ok) {
        const updatedNetwork = await response.json()
        onNetworkUpdated(updatedNetwork)
        toast({
          title: "Parameters Updated",
          description: "Network parameters have been successfully updated",
        })
        setHasChanges(false)
      } else {
        throw new Error("Failed to update parameters")
      }
    } catch (error) {
      toast({
        title: "Update Failed",
        description: "Failed to update network parameters",
        variant: "destructive",
      })
    } finally {
      setIsUpdating(false)
    }
  }

  const handleResetParameters = () => {
    setEmbeddingDim([network.embedding_dim])
    setBeta([network.beta])
    setLearningRate([network.learning_rate])
    setHasChanges(false)
  }

  // Camera control functions
  const handleZoomIn = () => {
    const newDistance = Math.max(cameraDistance[0] - 2, 5)
    setCameraDistance([newDistance])
    window.dispatchEvent(
      new CustomEvent("brainControl", {
        detail: { type: "zoom", value: newDistance },
      }),
    )
  }

  const handleZoomOut = () => {
    const newDistance = Math.min(cameraDistance[0] + 2, 30)
    setCameraDistance([newDistance])
    window.dispatchEvent(
      new CustomEvent("brainControl", {
        detail: { type: "zoom", value: newDistance },
      }),
    )
  }

  const handleResetCamera = () => {
    setCameraDistance([9])
    setRotationSpeed([2])
    setAutoRotate(true)
    window.dispatchEvent(
      new CustomEvent("brainControl", {
        detail: { type: "reset" },
      }),
    )
  }

  const handleToggleAutoRotate = () => {
    const newAutoRotate = !autoRotate
    setAutoRotate(newAutoRotate)
    window.dispatchEvent(
      new CustomEvent("brainControl", {
        detail: { type: "autoRotate", value: newAutoRotate },
      }),
    )
  }

  return (
    <Tabs defaultValue="network-controls" className="w-full">
      {" "}
      {/* Changed default value */}
      <TabsList className="grid w-full grid-cols-3">
        {/* Reordered tabs */}
        <TabsTrigger value="network-controls">
          <Zap className="w-4 h-4 mr-2" />
          Network Controls
        </TabsTrigger>
        <TabsTrigger value="parameters">
          <Settings className="w-4 h-4 mr-2" />
          Parameters
        </TabsTrigger>
        <TabsTrigger value="camera">
          <RotateCcw className="w-4 h-4 mr-2" />
          Camera
        </TabsTrigger>
      </TabsList>
      {/* New TabsContent for Network Controls */}
      <TabsContent value="network-controls" className="space-y-6 mt-6">
        <NetworkControls networkId={network.id} />
      </TabsContent>
      <TabsContent value="parameters" className="space-y-6 mt-6">
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Embedding Dimension</Label>
              <Badge variant="outline">{embeddingDim[0]}</Badge>
            </div>
            <Slider
              value={embeddingDim}
              onValueChange={setEmbeddingDim}
              max={512}
              min={32}
              step={32}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>32</span>
              <span>512</span>
              <span>1024</span>
              <span>2048</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Beta (Temperature)</Label>
              <Badge variant="outline">{beta[0]}</Badge>
            </div>
            <Slider value={beta} onValueChange={setBeta} max={50} min={1} step={1} className="w-full" />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>50</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Learning Rate</Label>
              <Badge variant="outline">{learningRate[0]}</Badge>
            </div>
            <Slider
              value={learningRate}
              onValueChange={setLearningRate}
              max={1}
              min={0.01}
              step={0.01}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0.01</span>
              <span>1.0</span>
            </div>
          </div>
        </div>

        <div className="flex space-x-2">
          <Button
            onClick={handleUpdateParameters}
            disabled={!hasChanges || isUpdating}
            className="flex-1 bg-blue-600 hover:bg-blue-700"
          >
            {isUpdating ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Updating...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </>
            )}
          </Button>
          <Button onClick={handleResetParameters} disabled={!hasChanges} variant="outline">
            Reset
          </Button>
        </div>
      </TabsContent>
      <TabsContent value="camera" className="space-y-6 mt-6">
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Camera Distance</Label>
              <Badge variant="outline">{cameraDistance[0]}</Badge>
            </div>
            <Slider
              value={cameraDistance}
              onValueChange={(value) => {
                setCameraDistance(value)
                window.dispatchEvent(
                  new CustomEvent("brainControl", {
                    detail: { type: "zoom", value: value[0] },
                  }),
                )
              }}
              max={30}
              min={5}
              step={1}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Rotation Speed</Label>
              <Badge variant="outline">{rotationSpeed[0]}x</Badge>
            </div>
            <Slider
              value={rotationSpeed}
              onValueChange={(value) => {
                setRotationSpeed(value)
                window.dispatchEvent(
                  new CustomEvent("brainControl", {
                    detail: { type: "rotationSpeed", value: value[0] },
                  }),
                )
              }}
              max={10}
              min={0}
              step={1}
              className="w-full"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button onClick={handleZoomIn} variant="outline">
            <ZoomIn className="w-4 h-4 mr-2" />
            Zoom In
          </Button>
          <Button onClick={handleZoomOut} variant="outline">
            <ZoomOut className="w-4 h-4 mr-2" />
            Zoom Out
          </Button>
          <Button
            onClick={handleToggleAutoRotate}
            variant="outline"
            className={autoRotate ? "bg-blue-50 border-blue-300" : ""}
          >
            <RotateCw className="w-4 h-4 mr-2" />
            Auto Rotate
          </Button>
          <Button onClick={handleResetCamera} variant="outline">
            <Home className="w-4 h-4 mr-2" />
            Reset View
          </Button>
        </div>
      </TabsContent>
    </Tabs>
  )
}
