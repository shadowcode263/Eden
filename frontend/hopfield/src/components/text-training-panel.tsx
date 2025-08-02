"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BookOpen, Sparkles, Loader2, Upload, FileText } from "lucide-react"
import { toast } from "@/hooks/use-toast"

interface TextTrainingPanelProps {
  networkId: number
  onAction: () => void
}

export function TextTrainingPanel({ networkId, onAction }: TextTrainingPanelProps) {
  const [learningText, setLearningText] = useState("")
  const [predictionPrompt, setPredictionPrompt] = useState("")
  const [predictionResult, setPredictionResult] = useState("")
  const [isLearning, setIsLearning] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [isUploadingBook, setIsUploadingBook] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleLearnText = async () => {
    if (!learningText.trim()) {
      toast({
        title: "Error",
        description: "Please enter some text to learn from.",
        variant: "destructive",
      })
      return
    }

    setIsLearning(true)
    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/learn/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          network_id: networkId,
          text_content: learningText,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        toast({
          title: "Learning Complete",
          description: `Successfully learned from ${learningText.length} characters of text.`,
        })
        setLearningText("")
        onAction()
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to learn from text")
      }
    } catch (error) {
      console.error("Error learning text:", error)
      toast({
        title: "Learning Failed",
        description: error instanceof Error ? error.message : "Failed to learn from text.",
        variant: "destructive",
      })
    } finally {
      setIsLearning(false)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Check if it's a text file
    if (!file.type.startsWith("text/") && !file.name.endsWith(".txt")) {
      toast({
        title: "Invalid File Type",
        description: "Please select a text file (.txt).",
        variant: "destructive",
      })
      return
    }

    // Check file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File Too Large",
        description: "Please select a file smaller than 10MB.",
        variant: "destructive",
      })
      return
    }

    setSelectedFile(file)
  }

  const handleUploadBook = async () => {
    if (!selectedFile) {
      toast({
        title: "No File Selected",
        description: "Please select a text file to upload.",
        variant: "destructive",
      })
      return
    }

    setIsUploadingBook(true)
    try {
      // Read file content
      const fileContent = await selectedFile.text()

      const response = await fetch("http://localhost:8000/api/brain/actions/learn-book/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          network_id: networkId,
          book_content: fileContent,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        toast({
          title: "Book Learning Complete",
          description: `Successfully learned from "${selectedFile.name}" (${fileContent.length} characters).`,
        })
        setSelectedFile(null)
        // Reset file input
        const fileInput = document.getElementById("book-file") as HTMLInputElement
        if (fileInput) fileInput.value = ""
        onAction()
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || errorData.detail || "Failed to learn from book")
      }
    } catch (error) {
      console.error("Error uploading book:", error)
      toast({
        title: "Book Learning Failed",
        description: error instanceof Error ? error.message : "Failed to learn from book.",
        variant: "destructive",
      })
    } finally {
      setIsUploadingBook(false)
    }
  }

  const handlePredictText = async () => {
    if (!predictionPrompt.trim()) {
      toast({
        title: "Error",
        description: "Please enter a prompt for prediction.",
        variant: "destructive",
      })
      return
    }

    setIsPredicting(true)
    try {
      const response = await fetch("http://localhost:8000/api/brain/actions/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          network_id: networkId,
          start_text: predictionPrompt,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        const continuation = data.story_continuation || data.prediction || data.continuation

        if (Array.isArray(continuation)) {
          setPredictionResult(continuation.join(" "))
        } else if (typeof continuation === "string") {
          setPredictionResult(continuation)
        } else {
          setPredictionResult("No prediction generated")
        }

        toast({
          title: "Prediction Complete",
          description: "Story continuation generated successfully.",
        })
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to generate prediction")
      }
    } catch (error) {
      console.error("Error predicting text:", error)
      toast({
        title: "Prediction Failed",
        description: error instanceof Error ? error.message : "Failed to generate prediction.",
        variant: "destructive",
      })
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="text" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="text">Learn Text</TabsTrigger>
          <TabsTrigger value="book">Upload Book</TabsTrigger>
          <TabsTrigger value="predict">Predict</TabsTrigger>
        </TabsList>

        <TabsContent value="text" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-primary" />
                Learn from Text
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="learning-text">Training Text</Label>
                <Textarea
                  id="learning-text"
                  placeholder="Enter text for the network to learn from..."
                  value={learningText}
                  onChange={(e) => setLearningText(e.target.value)}
                  rows={8}
                  className="resize-none"
                />
                <p className="text-xs text-muted-foreground">Characters: {learningText.length}</p>
              </div>
              <Button onClick={handleLearnText} disabled={isLearning || !learningText.trim()} className="w-full">
                {isLearning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Learning...
                  </>
                ) : (
                  <>
                    <BookOpen className="w-4 h-4 mr-2" />
                    Learn from Text
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="book" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                Upload Book File
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="book-file">Select Text File (.txt)</Label>
                <Input
                  id="book-file"
                  type="file"
                  accept=".txt,text/plain"
                  onChange={handleFileUpload}
                  className="cursor-pointer"
                />
                <p className="text-xs text-muted-foreground">
                  Upload a text file (max 10MB) for the network to learn from using self-supervised learning.
                </p>
              </div>

              {selectedFile && (
                <div className="p-3 bg-muted rounded-md border">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-primary" />
                    <span className="font-medium">{selectedFile.name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Size: {(selectedFile.size / 1024).toFixed(1)} KB</p>
                </div>
              )}

              <Button onClick={handleUploadBook} disabled={isUploadingBook || !selectedFile} className="w-full">
                {isUploadingBook ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing Book...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Learn from Book
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="predict" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary" />
                Predict Story Continuation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="prediction-prompt">Story Prompt</Label>
                <Input
                  id="prediction-prompt"
                  placeholder="Enter a story beginning..."
                  value={predictionPrompt}
                  onChange={(e) => setPredictionPrompt(e.target.value)}
                />
              </div>
              <Button
                onClick={handlePredictText}
                disabled={isPredicting || !predictionPrompt.trim()}
                className="w-full"
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Generate Continuation
                  </>
                )}
              </Button>

              {predictionResult && (
                <div className="space-y-2">
                  <Label>Generated Continuation</Label>
                  <div className="p-3 bg-muted rounded-md border">
                    <p className="text-sm whitespace-pre-wrap">{predictionResult}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
