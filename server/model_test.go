package server

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/template"
)

func TestToolToken(t *testing.T) {
	cases := []struct {
		input string
		want  string
		ok    bool
	}{
		{
			input: "Action: ```json",
			want:  "Action:",
			ok:    true,
		},
		{
			input: "functools[",
			want:  "functools",
			ok:    true,
		},
		{
			input: "Hello, world! <tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[tool_call] <tool_call>",
			want:  "[tool_call]",
			ok:    true,
		},
		{
			input: "<tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[tool_call] <",
			want:  "[tool_call]",
			ok:    true,
		},
		{
			input: "> <tool_call>",
			want:  "<tool_call>",
			ok:    true,
		},
		{
			input: "[TOOL_CALL] [",
			want:  "[TOOL_CALL]",
			ok:    true,
		},
		{
			input: "[TOOL_CALL][",
			want:  "[TOOL_CALL]",
			ok:    true,
		},
		{
			input: "<|tool_call|>",
			want:  "<|tool_call|>",
			ok:    true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.input, func(t *testing.T) {
			tmpl, err := template.Parse(tt.input)
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}
			m := &Model{Template: tmpl}
			fmt.Println("m.Template", m.Template.String())
			got, ok := ToolToken(tt.input)
			if got != tt.want {
				t.Errorf("ToolToken(%q) = %q; want %q", tt.input, got, tt.want)
			}
			if ok != tt.ok {
				t.Errorf("ToolToken(%q) = %v; want %v", tt.input, ok, tt.ok)
			}
		})
	}
}
